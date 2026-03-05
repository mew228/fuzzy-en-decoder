"""
======================================================
  Encoder-Decoder CNN for Grayscale Image Denoising
======================================================
Dataset  : MNIST (28x28 grayscale handwritten digits)
Noise    : Gaussian noise added synthetically
Model    : Convolutional Autoencoder (Encoder–Decoder)
Loss     : Mean Squared Error (MSE)
======================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE THE DATASET
# ──────────────────────────────────────────────────────────────────────────────
def load_and_prepare_data(noise_factor: float = 0.4):
    """Download MNIST, normalise to [0,1], and add Gaussian noise."""
    print("\n[1/4] Loading MNIST dataset …")
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    # Normalise & add channel dimension  →  (N, 28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test  = np.expand_dims(x_test,  axis=-1)

    # Synthetic Gaussian noise
    rng = np.random.default_rng(seed=42)
    x_train_noisy = x_train + noise_factor * rng.standard_normal(x_train.shape)
    x_test_noisy  = x_test  + noise_factor * rng.standard_normal(x_test.shape)

    # Clip to keep values in [0, 1]
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy  = np.clip(x_test_noisy,  0.0, 1.0)

    print(f"  Train samples : {len(x_train):,}   |  Test samples : {len(x_test):,}")
    print(f"  Image shape   : {x_train.shape[1:]}   |  Noise factor : {noise_factor}")
    return x_train, x_train_noisy, x_test, x_test_noisy


# ──────────────────────────────────────────────────────────────────────────────
# 2. BUILD THE ENCODER–DECODER MODEL
# ──────────────────────────────────────────────────────────────────────────────
def build_autoencoder(input_shape=(28, 28, 1)) -> models.Model:
    """
    Convolutional Autoencoder

    Encoder
    -------
      Conv(32, 3x3, ReLU) → MaxPool(2x2)   →  14×14×32
      Conv(64, 3x3, ReLU) → MaxPool(2x2)   →   7× 7×64
      Conv(128, 3x3, ReLU)                  →   7× 7×128  (latent)

    Decoder
    -------
      Conv(128, 3x3, ReLU) + UpSampling(2)  →  14×14×128
      Conv(64,  3x3, ReLU) + UpSampling(2)  →  28×28×64
      Conv(32,  3x3, ReLU)                   →  28×28×32
      Conv(1,   3x3, Sigmoid)                →  28×28× 1  (output)
    """
    inputs = Input(shape=input_shape, name="noisy_input")

    # ── Encoder ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)          # 14×14

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool2")(x)          # 7×7

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="latent")(x)
    # → latent representation: 7×7×128

    # ── Decoder ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="dec_conv1")(x)
    x = layers.UpSampling2D((2, 2), name="dec_up1")(x)                            # 14×14

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="dec_conv2")(x)
    x = layers.UpSampling2D((2, 2), name="dec_up2")(x)                            # 28×28

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_conv3")(x)
    outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="clean_output")(x)

    model = models.Model(inputs, outputs, name="ConvAutoencoder")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 3. COMPILE & TRAIN
# ──────────────────────────────────────────────────────────────────────────────
def train_model(model, x_train_noisy, x_train, x_test_noisy, x_test,
                epochs: int = 15, batch_size: int = 128):
    """Compile with MSE loss and train the autoencoder."""
    print("\n[3/4] Compiling and training the model …")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    history = model.fit(
        x_train_noisy, x_train,          # input=noisy, target=clean
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
        verbose=1,
    )
    print("  Training complete.")
    return history


# ──────────────────────────────────────────────────────────────────────────────
# 4. VISUALISE RESULTS
# ──────────────────────────────────────────────────────────────────────────────
def visualise_results(model, x_test, x_test_noisy,
                       n_samples: int = 10, save_path: str = "denoising_results.png"):
    """Plot original  |  noisy  |  reconstructed images side-by-side."""
    print("\n[4/4] Generating visualisation …")
    reconstructed = model.predict(x_test_noisy[:n_samples])

    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 1.5, 4.5))
    fig.patch.set_facecolor("#0d1117")

    row_labels = ["Original (Clean)", "Noisy Input", "Reconstructed"]
    images     = [x_test[:n_samples], x_test_noisy[:n_samples], reconstructed]
    border_colors = ["#4caf50", "#f44336", "#2196f3"]

    for row, (label, imgs, bcolor) in enumerate(zip(row_labels, images, border_colors)):
        for col in range(n_samples):
            ax = axes[row, col]
            ax.imshow(imgs[col].squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(bcolor)
                spine.set_linewidth(2)
            if col == 0:
                ax.set_ylabel(label, fontsize=8, color=bcolor,
                              fontweight="bold", rotation=90,
                              labelpad=8, va="center")

    fig.suptitle("Encoder–Decoder CNN  ·  Image Denoising (MNIST)",
                 fontsize=13, color="white", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Figure saved → {save_path}")


def plot_training_history(history, save_path: str = "training_history.png"):
    """Plot MSE training & validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.plot(history.history["loss"],     label="Train MSE", color="#4caf50", linewidth=2)
    ax.plot(history.history["val_loss"], label="Val   MSE", color="#2196f3",
            linewidth=2, linestyle="--")

    ax.set_title("Training & Validation Loss (MSE)",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("MSE Loss", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", labelcolor="white")
    ax.grid(color="#30363d", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"  Loss curve saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Suppress verbose TF logs (optional)
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # 1. Data
    x_train, x_train_noisy, x_test, x_test_noisy = load_and_prepare_data(noise_factor=0.4)

    # 2. Model
    print("\n[2/4] Building Encoder–Decoder model …")
    autoencoder = build_autoencoder(input_shape=(28, 28, 1))

    # 3. Train
    history = train_model(
        autoencoder,
        x_train_noisy, x_train,
        x_test_noisy,  x_test,
        epochs=15,
        batch_size=128,
    )

    # 4. Visualise
    visualise_results(autoencoder, x_test, x_test_noisy,
                      n_samples=10, save_path="denoising_results.png")
    plot_training_history(history, save_path="training_history.png")

    # Final evaluation
    test_loss, test_mae = autoencoder.evaluate(x_test_noisy, x_test, verbose=0)
    print(f"\n  ✅  Test MSE : {test_loss:.6f}   |   Test MAE : {test_mae:.6f}")
