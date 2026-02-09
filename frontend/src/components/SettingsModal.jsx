import { useState, useEffect } from 'react'

export default function SettingsModal({ onClose, apiBase }) {
    const [config, setConfig] = useState({
        llm_mode: 'online',
        gemini_api_key_configured: false,
        ollama_base_url: '',
        ollama_model: ''
    })
    const [loading, setLoading] = useState(true)
    const [saving, setSaving] = useState(false)
    const [apiKey, setApiKey] = useState('')
    const [error, setError] = useState(null)
    const [success, setSuccess] = useState(false)

    // Load current config on mount
    useEffect(() => {
        fetch(`${apiBase}/config`)
            .then(res => res.json())
            .then(data => {
                setConfig(data)
                setLoading(false)
            })
            .catch(err => {
                console.error("Failed to load config:", err)
                setError("Failed to load configuration")
                setLoading(false)
            })
    }, [apiBase])

    const handleSave = async () => {
        setSaving(true)
        setError(null)
        setSuccess(false)

        try {
            const payload = {
                llm_mode: config.llm_mode,
                ollama_base_url: config.ollama_base_url,
                ollama_model: config.ollama_model
            }

            // Only send API key if entered
            if (apiKey.trim()) {
                payload.gemini_api_key = apiKey.trim()
            }

            const res = await fetch(`${apiBase}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || "Failed to save settings")
            }

            const updated = await res.json()
            setConfig(updated)
            setApiKey('') // Clear input after save
            setSuccess(true)

            // Close after short delay if successful
            setTimeout(() => {
                onClose()
            }, 1000)

        } catch (err) {
            setError(err.message)
        } finally {
            setSaving(false)
        }
    }

    if (loading) return null

    return (
        <div className="modal-backdrop">
            <div className="modal-container">
                <div className="modal-header">
                    <h2>⚙️ Settings</h2>
                    <button className="btn-close" onClick={onClose}>×</button>
                </div>

                <div className="modal-body">
                    {error && <div className="alert error">{error}</div>}
                    {success && <div className="alert success">Settings saved successfully! ✔️</div>}

                    <div className="form-group">
                        <label>LLM Mode</label>
                        <div className="mode-toggle">
                            <button
                                className={`mode-btn ${config.llm_mode === 'online' ? 'active' : ''}`}
                                onClick={() => setConfig({ ...config, llm_mode: 'online' })}
                            >
                                ☁️ Online (Gemini)
                            </button>
                            <button
                                className={`mode-btn ${config.llm_mode === 'offline' ? 'active' : ''}`}
                                onClick={() => setConfig({ ...config, llm_mode: 'offline' })}
                            >
                                🏠 Offline (Ollama)
                            </button>
                        </div>
                    </div>

                    {config.llm_mode === 'online' && (
                        <div className="form-group fade-in">
                            <label>Gemini API Key</label>
                            <div className="input-wrapper">
                                <input
                                    type="password"
                                    placeholder={config.gemini_api_key_configured ? "•••••••• (Configured)" : "Enter API Key"}
                                    value={apiKey}
                                    onChange={e => setApiKey(e.target.value)}
                                />
                                {config.gemini_api_key_configured && !apiKey && (
                                    <span className="badge-configured">Active</span>
                                )}
                            </div>
                            <p className="help-text">
                                Get key from <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noreferrer">Google AI Studio</a>
                            </p>
                        </div>
                    )}

                    {config.llm_mode === 'offline' && (
                        <div className="fade-in">
                            <div className="form-group">
                                <label>Ollama Base URL</label>
                                <input
                                    type="text"
                                    value={config.ollama_base_url}
                                    onChange={e => setConfig({ ...config, ollama_base_url: e.target.value })}
                                    placeholder="http://localhost:11434"
                                />
                            </div>
                            <div className="form-group">
                                <label>Ollama Model</label>
                                <input
                                    type="text"
                                    value={config.ollama_model}
                                    onChange={e => setConfig({ ...config, ollama_model: e.target.value })}
                                    placeholder="llama3, mistral, etc."
                                />
                            </div>
                            <p className="help-text warning">
                                ⚠️ Ensure Ollama is running (`ollama serve`) and model is pulled.
                            </p>
                        </div>
                    )}
                </div>

                <div className="modal-footer">
                    <button className="btn-cancel" onClick={onClose} disabled={saving}>Cancel</button>
                    <button className="btn-save" onClick={handleSave} disabled={saving}>
                        {saving ? 'Saving...' : 'Save Settings'}
                    </button>
                </div>
            </div>
        </div>
    )
}
