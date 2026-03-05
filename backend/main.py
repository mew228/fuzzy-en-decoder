"""
FastAPI Backend for Encoder-Decoder Image Denoising
-----------------------------------------------------
Endpoints:
  GET  /            → health check + model status
  GET  /health      → Railway health check
  POST /train       → trigger model training (async)
  POST /denoise     → upload an image → returns original, noisy, denoised as base64
  GET  /sample      → returns a random MNIST sample result
  GET  /status      → returns training progress / status
"""

import io
import os
import base64
import threading
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Suppress TF logs ──────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, Input

# ── Global state ──────────────────────────────────────────────────────────────
MODEL_PATH = Path("autoencoder_model.keras")
autoencoder = None          # Keras model
train_status = {
    "status": "idle",       # idle | training | done | error
    "epoch": 0,
    "total_epochs": 5,
    "val_loss": None,
    "message": "Model not trained yet. Call /train to start.",
}
x_test_clean = None
x_test_noisy = None
NOISE_FACTOR = 0.4


# ── Model definition (optimized 16→32→64 + BatchNorm) ────────────────────────
def build_autoencoder(input_shape=(28, 28, 1)):
    inputs = Input(shape=input_shape)

    # Encoder — lightweight for CPU deployment
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, padding="same")(x)          # 14×14

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, padding="same")(x)          # 7×7

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)                      # latent 7×7×64

    # Decoder
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)                          # 14×14

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)                          # 28×28

    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    return models.Model(inputs, outputs, name="ConvAutoencoder")


# ── Data loader ───────────────────────────────────────────────────────────────
def load_mnist():
    global x_test_clean, x_test_noisy
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test,  -1)

    rng = np.random.default_rng(42)
    x_train_noisy = np.clip(
        x_train + NOISE_FACTOR * rng.standard_normal(x_train.shape).astype("float32"),
        0, 1,
    ).astype("float32")
    x_test_noisy_ = np.clip(
        x_test + NOISE_FACTOR * rng.standard_normal(x_test.shape).astype("float32"),
        0, 1,
    ).astype("float32")

    x_test_clean = x_test
    x_test_noisy = x_test_noisy_
    return x_train, x_train_noisy, x_test, x_test_noisy_


# ── Training thread ───────────────────────────────────────────────────────────
class TrainCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_status["epoch"] = epoch + 1
        train_status["val_loss"] = round(float(logs.get("val_loss", 0)), 6)
        train_status["message"] = (
            f"Epoch {epoch+1}/{train_status['total_epochs']} — "
            f"val_loss: {train_status['val_loss']:.6f}"
        )


def _train_worker(epochs: int):
    global autoencoder
    try:
        train_status["status"] = "training"
        train_status["epoch"] = 0
        train_status["message"] = "Loading MNIST data…"

        x_train, x_train_noisy, x_test, x_test_noisy_ = load_mnist()

        train_status["message"] = "Building model…"
        autoencoder = build_autoencoder()

        # Enable JIT compile (XLA) if available
        try:
            autoencoder.compile(
                optimizer="adam", loss="mse", metrics=["mae"], jit_compile=True,
            )
        except Exception:
            autoencoder.compile(optimizer="adam", loss="mse", metrics=["mae"])

        train_status["total_epochs"] = epochs
        train_status["message"] = "Preparing fast data pipelines…"

        # Optimize data pipelines with tf.data
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train_noisy, x_train))
            .cache()
            .shuffle(10000)
            .batch(512)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((x_test_noisy_, x_test))
            .batch(512)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        train_status["message"] = "Training started (optimized CPU)…"

        autoencoder.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[TrainCallback()],
            verbose=0,
        )
        autoencoder.save(str(MODEL_PATH))
        train_status["status"] = "done"
        train_status["message"] = "Training complete! Model ready."
    except Exception as exc:
        train_status["status"] = "error"
        train_status["message"] = f"Training failed: {exc}"


# ── Lifespan (replaces deprecated on_event) ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load saved model on startup if available."""
    global autoencoder
    if MODEL_PATH.exists():
        try:
            autoencoder = tf.keras.models.load_model(str(MODEL_PATH))
            load_mnist()
            train_status["status"] = "done"
            train_status["message"] = "Pre-trained model loaded from disk."
        except Exception as e:
            train_status["message"] = f"Could not load saved model: {e}"
    yield  # app runs here
    # shutdown: nothing to clean up


# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Denoising Autoencoder API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS: allow any origin in dev; in production set
# ALLOWED_ORIGINS="https://your-app.vercel.app"
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = (
    [o.strip() for o in _raw_origins.split(",")]
    if _raw_origins != "*"
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Image helpers ─────────────────────────────────────────────────────────────
def arr_to_b64(arr_2d: np.ndarray) -> str:
    """Convert a (H,W) float32 array [0,1] to a PNG base64 string."""
    img_uint8 = (np.clip(arr_2d, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")
    pil_img = pil_img.resize((224, 224), Image.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def preprocess_upload(file_bytes: bytes) -> np.ndarray:
    """Read uploaded image → grayscale 28×28 numpy array."""
    img = Image.open(io.BytesIO(file_bytes)).convert("L").resize((28, 28))
    arr = np.array(img, dtype="float32") / 255.0
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app": "Denoising Autoencoder API",
        "version": "2.0.0",
        "model_ready": autoencoder is not None,
    }


@app.get("/health")
def health():
    """Railway / uptime health check."""
    return {"status": "ok"}


@app.get("/status")
def get_status():
    return train_status


@app.post("/train")
def start_training(epochs: int = Query(default=5, ge=1, le=50)):
    if train_status["status"] == "training":
        return {"message": "Training already in progress."}
    t = threading.Thread(target=_train_worker, args=(epochs,), daemon=True)
    t.start()
    return {"message": f"Training started for {epochs} epochs. Poll /status for progress."}


@app.post("/denoise")
async def denoise_image(
    file: UploadFile = File(...),
    noise_factor: float = Query(default=0.4, ge=0.0, le=1.0),
):
    if autoencoder is None:
        raise HTTPException(status_code=503, detail="Model not ready. Call /train first.")

    file_bytes = await file.read()
    try:
        clean_arr = preprocess_upload(file_bytes)  # (28,28)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # Add noise
    rng = np.random.default_rng()
    noisy_arr = np.clip(
        clean_arr + noise_factor * rng.standard_normal(clean_arr.shape).astype("float32"),
        0, 1,
    ).astype("float32")

    # Predict — use __call__ for single-image speed (faster than .predict)
    input_tensor = tf.constant(noisy_arr.reshape(1, 28, 28, 1))
    reconstructed = autoencoder(input_tensor, training=False).numpy()[0, :, :, 0]

    return JSONResponse({
        "original":      arr_to_b64(clean_arr),
        "noisy":         arr_to_b64(noisy_arr),
        "reconstructed": arr_to_b64(reconstructed),
        "noise_factor":  noise_factor,
    })


@app.get("/sample")
def get_sample(noise_factor: float = Query(default=0.4, ge=0.0, le=1.0)):
    if autoencoder is None:
        raise HTTPException(status_code=503, detail="Model not ready. Call /train first.")
    if x_test_clean is None:
        raise HTTPException(status_code=503, detail="MNIST data not loaded yet.")

    idx = int(np.random.randint(0, len(x_test_clean)))
    clean_arr = x_test_clean[idx, :, :, 0]

    rng = np.random.default_rng()
    noisy_arr = np.clip(
        clean_arr + noise_factor * rng.standard_normal(clean_arr.shape).astype("float32"),
        0, 1,
    ).astype("float32")

    # Use __call__ for speed
    input_tensor = tf.constant(noisy_arr.reshape(1, 28, 28, 1))
    reconstructed = autoencoder(input_tensor, training=False).numpy()[0, :, :, 0]

    return JSONResponse({
        "original":      arr_to_b64(clean_arr),
        "noisy":         arr_to_b64(noisy_arr),
        "reconstructed": arr_to_b64(reconstructed),
        "noise_factor":  noise_factor,
        "sample_index":  idx,
    })
