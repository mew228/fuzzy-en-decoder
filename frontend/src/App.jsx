import { useState, useRef, useCallback, useEffect } from 'react'
import './index.css'

// In local dev, Vite proxies /api → localhost:8000.
// In production (Vercel), set VITE_API_URL to your Railway backend URL.
const API = import.meta.env.VITE_API_URL ?? '/api'

// ── Hooks ─────────────────────────────────────────────────────────────────────
function useInterval(cb, delay) {
    const savedCb = useRef(cb)
    useEffect(() => { savedCb.current = cb }, [cb])
    useEffect(() => {
        if (delay == null) return
        const id = setInterval(() => savedCb.current(), delay)
        return () => clearInterval(id)
    }, [delay])
}

// Safe fetch wrapper with timeout
async function safeFetch(url, options = {}) {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 30000)
    try {
        const res = await fetch(url, { ...options, signal: controller.signal })
        clearTimeout(timeout)
        return res
    } catch (e) {
        clearTimeout(timeout)
        throw e
    }
}

// ── Toast notification system ─────────────────────────────────────────────────
function Toast({ message, type, onClose }) {
    useEffect(() => {
        const timer = setTimeout(onClose, 4000)
        return () => clearTimeout(timer)
    }, [onClose])

    return (
        <div className={`toast toast-${type}`}>
            <span>{type === 'error' ? '❌' : type === 'success' ? '✅' : 'ℹ️'}</span>
            <span>{message}</span>
            <button className="toast-close" onClick={onClose}>×</button>
        </div>
    )
}

function ToastContainer({ toasts, removeToast }) {
    return (
        <div className="toast-container">
            {toasts.map((t) => (
                <Toast key={t.id} {...t} onClose={() => removeToast(t.id)} />
            ))}
        </div>
    )
}

// ── Sub-components ────────────────────────────────────────────────────────────
function StatusBanner({ status, retrying }) {
    const icon = status.status === 'done' ? '✅'
        : status.status === 'error' ? '❌'
            : status.status === 'training' ? null
                : 'ℹ️'
    const pct = status.status === 'training'
        ? Math.round((status.epoch / status.total_epochs) * 100)
        : status.status === 'done' ? 100 : 0

    return (
        <div className={`status-banner ${status.status}`}>
            {status.status === 'training'
                ? <div className="spinner" />
                : <span className="status-icon">{icon}</span>
            }
            <div style={{ flex: 1 }}>
                <div>
                    {retrying
                        ? '🔄 Connecting to backend… (cold-starting)'
                        : status.message
                    }
                </div>
                {status.status === 'training' && (
                    <div className="progress-wrap" style={{ marginTop: 8 }}>
                        <div className="progress-bar" style={{ width: `${pct}%` }} />
                    </div>
                )}
            </div>
            {status.status === 'training' && (
                <span className="pct-label">{pct}%</span>
            )}
        </div>
    )
}

function ImageCard({ src, label, dotClass, note }) {
    if (!src) return null
    return (
        <div className="image-card">
            <div className="img-header">
                <span className={`img-dot ${dotClass}`} />
                <span>{label}</span>
                {note && <small>{note}</small>}
            </div>
            <img src={`data:image/png;base64,${src}`} alt={label} loading="lazy" />
        </div>
    )
}

function DownloadButton({ base64Src, filename }) {
    const download = useCallback(() => {
        const link = document.createElement('a')
        link.href = `data:image/png;base64,${base64Src}`
        link.download = filename
        link.click()
    }, [base64Src, filename])

    return (
        <button className="btn btn-secondary btn-sm" onClick={download} title="Download image">
            💾 Download
        </button>
    )
}

function ResultsPanel({ result, noiseFactor }) {
    if (!result) return null
    return (
        <div className="card">
            <div className="card-title">
                Results
                {result.sample_index != null && (
                    <span className="card-title-note">Sample #{result.sample_index}</span>
                )}
            </div>
            <div className="results-grid">
                <ImageCard src={result.original} label="Original (Clean)" dotClass="green" note="ground truth" />
                <ImageCard src={result.noisy} label="Noisy Input" dotClass="red" note={`σ = ${noiseFactor.toFixed(2)}`} />
                <ImageCard src={result.reconstructed} label="Reconstructed" dotClass="blue" note="model output" />
            </div>
            <div className="download-row">
                <DownloadButton base64Src={result.original} filename="original.png" />
                <DownloadButton base64Src={result.noisy} filename="noisy.png" />
                <DownloadButton base64Src={result.reconstructed} filename="denoised.png" />
            </div>
        </div>
    )
}

function UploadTab({ modelReady, noiseFactor, setNoiseFactor, addToast }) {
    const [file, setFile] = useState(null)
    const [drag, setDrag] = useState(false)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)

    const handleFile = useCallback((f) => {
        if (!f) return
        setFile(f)
        setResult(null)
    }, [])

    const onDrop = useCallback((e) => {
        e.preventDefault()
        setDrag(false)
        const f = e.dataTransfer.files[0]
        if (f) handleFile(f)
    }, [handleFile])

    const submit = async () => {
        if (!file) return
        setLoading(true)
        setResult(null)
        const fd = new FormData()
        fd.append('file', file)
        try {
            const res = await safeFetch(`${API}/denoise?noise_factor=${noiseFactor}`, {
                method: 'POST',
                body: fd,
            })
            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || 'Server error')
            }
            const data = await res.json()
            setResult(data)
            addToast('Image denoised successfully!', 'success')
        } catch (e) {
            addToast(e.message || 'Network error', 'error')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div>
            <div className="card">
                <div className="card-title">Upload Image</div>
                <div
                    className={`upload-zone ${drag ? 'drag-over' : ''}`}
                    onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
                    onDragLeave={() => setDrag(false)}
                    onDrop={onDrop}
                >
                    <input
                        type="file" accept="image/*"
                        onChange={(e) => handleFile(e.target.files[0])}
                        id="file-upload"
                    />
                    <div className="upload-icon">🖼️</div>
                    <p><strong>Click or drag</strong> an image here<br />Any image — it'll be converted to 28×28 grayscale</p>
                    {file && <div className="file-selected">📎 {file.name}</div>}
                </div>

                <div className="slider-group">
                    <div className="slider-label">
                        <span>Noise Factor</span>
                        <span>{noiseFactor.toFixed(2)}</span>
                    </div>
                    <input
                        type="range" min="0.05" max="0.95" step="0.05"
                        value={noiseFactor}
                        onChange={(e) => setNoiseFactor(parseFloat(e.target.value))}
                    />
                </div>

                <div className="action-row">
                    <button
                        className="btn btn-primary"
                        style={{ flex: 1 }}
                        disabled={!file || !modelReady || loading}
                        onClick={submit}
                        id="denoise-btn"
                    >
                        {loading
                            ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Processing…</>
                            : '⚡ Denoise Image'
                        }
                    </button>
                </div>
            </div>

            <ResultsPanel result={result} noiseFactor={noiseFactor} />
        </div>
    )
}

function SampleTab({ modelReady, noiseFactor, setNoiseFactor, addToast }) {
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)

    const fetchSample = async () => {
        setLoading(true)
        setResult(null)
        try {
            const res = await safeFetch(`${API}/sample?noise_factor=${noiseFactor}`)
            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || 'Server error')
            }
            const data = await res.json()
            setResult(data)
            addToast(`Sample #${data.sample_index} loaded`, 'success')
        } catch (e) {
            addToast(e.message || 'Network error', 'error')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div>
            <div className="card">
                <div className="card-title">Random MNIST Sample</div>
                <p className="card-desc">
                    Pick a random digit from the MNIST test set, add Gaussian noise, and run it through the autoencoder.
                    No image upload needed.
                </p>

                <div className="slider-group">
                    <div className="slider-label">
                        <span>Noise Factor</span>
                        <span>{noiseFactor.toFixed(2)}</span>
                    </div>
                    <input
                        type="range" min="0.05" max="0.95" step="0.05"
                        value={noiseFactor}
                        onChange={(e) => setNoiseFactor(parseFloat(e.target.value))}
                    />
                </div>

                <button
                    className="btn btn-success"
                    style={{ marginTop: 18, width: '100%' }}
                    disabled={!modelReady || loading}
                    onClick={fetchSample}
                    id="sample-btn"
                >
                    {loading
                        ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Loading…</>
                        : '🎲 Random Sample'
                    }
                </button>
            </div>

            <ResultsPanel result={result} noiseFactor={noiseFactor} />
        </div>
    )
}

function TrainTab({ status, setStatus, addToast }) {
    const [epochs, setEpochs] = useState(5)
    const [polling, setPolling] = useState(false)

    // Resume polling if we're already training on mount
    useEffect(() => {
        if (status.status === 'training') setPolling(true)
    }, [])

    useInterval(async () => {
        if (!polling) return
        try {
            const res = await safeFetch(`${API}/status`)
            const s = await res.json()
            setStatus(s)
            if (s.status === 'done') {
                setPolling(false)
                addToast('Training complete! Model is ready.', 'success')
            } else if (s.status === 'error') {
                setPolling(false)
                addToast(`Training failed: ${s.message}`, 'error')
            }
        } catch {
            // Silently retry on network error — don't stop polling
        }
    }, polling ? 1500 : null)

    const startTrain = async () => {
        try {
            const res = await safeFetch(`${API}/train?epochs=${epochs}`, { method: 'POST' })
            const data = await res.json()
            setStatus((p) => ({ ...p, status: 'training', epoch: 0, total_epochs: epochs, message: data.message }))
            setPolling(true)
            addToast(`Training started for ${epochs} epochs`, 'info')
        } catch (e) {
            addToast(e.message || 'Could not start training', 'error')
        }
    }

    return (
        <div className="card">
            <div className="card-title">Model Training</div>
            <p className="card-desc">
                Train the Encoder–Decoder CNN on 60,000 MNIST digits (Gaussian noise added automatically).
                The model is saved to disk and reloaded automatically on server restart.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {/* Architecture diagram — matches actual backend model */}
                <div className="arch-block">
                    <div className="arch-node enc">Input 28×28</div>
                    <span className="arch-arrow">→</span>
                    <div className="arch-node enc">Conv16 + BN + Pool</div>
                    <span className="arch-arrow">→</span>
                    <div className="arch-node enc">Conv32 + BN + Pool</div>
                    <span className="arch-arrow">→</span>
                    <div className="arch-node lat">Conv64 + BN (Latent 7×7)</div>
                    <span className="arch-arrow">→</span>
                    <div className="arch-node dec">Conv64 + BN + Up</div>
                    <span className="arch-arrow">→</span>
                    <div className="arch-node dec">Conv32 + BN + Up</div>
                    <span className="arch-arrow">→</span>
                    <div className="arch-node dec">Conv16 → Output</div>
                </div>

                <div className="epoch-group">
                    <label htmlFor="epoch-input">Epochs:</label>
                    <input
                        id="epoch-input"
                        type="number" min="1" max="50"
                        className="epoch-input"
                        value={epochs}
                        onChange={(e) => setEpochs(Math.min(50, Math.max(1, +e.target.value)))}
                    />
                    <div className="metrics" style={{ flex: 1 }}>
                        {status.val_loss != null && (
                            <div className="metric-chip">Val MSE: <strong>{status.val_loss}</strong></div>
                        )}
                        {status.epoch > 0 && (
                            <div className="metric-chip">Epoch: <strong>{status.epoch}/{status.total_epochs}</strong></div>
                        )}
                    </div>
                </div>

                <button
                    className="btn btn-primary"
                    disabled={status.status === 'training'}
                    onClick={startTrain}
                    style={{ alignSelf: 'flex-start', paddingInline: 28 }}
                    id="train-btn"
                >
                    {status.status === 'training'
                        ? <><span className="spinner" style={{ width: 14, height: 14 }} /> Training…</>
                        : status.status === 'done' ? '🔄 Re-train Model' : '🚀 Start Training'
                    }
                </button>
            </div>
        </div>
    )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
    const [tab, setTab] = useState('sample')
    const [noiseFactor, setNoiseFactor] = useState(0.4)
    const [retrying, setRetrying] = useState(false)
    const [toasts, setToasts] = useState([])
    const toastId = useRef(0)
    const [status, setStatus] = useState({
        status: 'idle', epoch: 0, total_epochs: 5, val_loss: null,
        message: 'Connecting to backend…',
    })

    const addToast = useCallback((message, type = 'info') => {
        const id = ++toastId.current
        setToasts((prev) => [...prev.slice(-3), { id, message, type }])
    }, [])

    const removeToast = useCallback((id) => {
        setToasts((prev) => prev.filter((t) => t.id !== id))
    }, [])

    // Initial status check with retry for cold-starting backends
    useEffect(() => {
        let retryCount = 0
        let cancelled = false

        const fetchStatus = async () => {
            try {
                const res = await safeFetch(`${API}/status`)
                if (!cancelled) {
                    const data = await res.json()
                    setStatus(data)
                    setRetrying(false)
                }
            } catch {
                if (!cancelled && retryCount < 10) {
                    retryCount++
                    setRetrying(true)
                    setStatus((p) => ({
                        ...p,
                        message: `Connecting to backend… (attempt ${retryCount}/10)`,
                    }))
                    setTimeout(fetchStatus, 3000)
                } else if (!cancelled) {
                    setRetrying(false)
                    setStatus((p) => ({
                        ...p,
                        message: '⚠️ Cannot reach backend — make sure FastAPI is running',
                    }))
                }
            }
        }
        fetchStatus()
        return () => { cancelled = true }
    }, [])

    const modelReady = status.status === 'done'

    return (
        <div className="app-wrapper">
            <ToastContainer toasts={toasts} removeToast={removeToast} />

            <header className="hero">
                <div className="hero-glow" />
                <div className="hero-badge">
                    <span className="dot" />
                    TensorFlow · CNN · MSE Loss
                </div>
                <h1>Encoder–Decoder<br />Image Denoiser</h1>
                <p>Upload a grayscale image or pick a random MNIST sample.
                    The convolutional autoencoder removes Gaussian noise and reconstructs the clean image.</p>
            </header>

            <StatusBanner status={status} retrying={retrying} />

            <div className="tabs">
                {[['train', '⚙️ Train'], ['sample', '🎲 Sample'], ['upload', '📤 Upload']].map(([k, label]) => (
                    <button
                        key={k}
                        className={`tab-btn ${tab === k ? 'active' : ''}`}
                        onClick={() => setTab(k)}
                        id={`tab-${k}`}
                    >
                        {label}
                    </button>
                ))}
            </div>

            {tab === 'train' && <TrainTab status={status} setStatus={setStatus} addToast={addToast} />}
            {tab === 'sample' && (
                modelReady
                    ? <SampleTab modelReady={modelReady} noiseFactor={noiseFactor} setNoiseFactor={setNoiseFactor} addToast={addToast} />
                    : <div className="card empty">
                        <div className="empty-icon">🤖</div>
                        <p>Train the model first using the <strong>⚙️ Train</strong> tab.</p>
                    </div>
            )}
            {tab === 'upload' && (
                modelReady
                    ? <UploadTab modelReady={modelReady} noiseFactor={noiseFactor} setNoiseFactor={setNoiseFactor} addToast={addToast} />
                    : <div className="card empty">
                        <div className="empty-icon">🤖</div>
                        <p>Train the model first using the <strong>⚙️ Train</strong> tab.</p>
                    </div>
            )}

            <footer className="footer">
                <div className="footer-inner">
                    <span>Encoder–Decoder CNN · TensorFlow/Keras · MSE Loss · MNIST</span>
                    <span className="footer-sep">·</span>
                    <span>FastAPI + Vite + React</span>
                </div>
            </footer>
        </div>
    )
}
