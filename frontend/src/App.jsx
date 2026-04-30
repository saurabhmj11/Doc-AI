import { useState, useRef, useEffect } from 'react'
import SettingsModal from './components/SettingsModal'
import Login from './components/Login'
import api from './services/api'

function App() {
    // State management
    const [user, setUser] = useState(null)
    const [authChecked, setAuthChecked] = useState(false)
    const [files, setFiles] = useState([])
    const [processedDocs, setProcessedDocs] = useState([])
    const [uploading, setUploading] = useState(false)
    const [messages, setMessages] = useState([])
    const [question, setQuestion] = useState('')
    const [asking, setAsking] = useState(false)
    const [activeTab, setActiveTab] = useState('chat')
    const [extraction, setExtraction] = useState(null)
    const [extracting, setExtracting] = useState(false)
    const [pipelineStage, setPipelineStage] = useState(null)
    const [showArchitecture, setShowArchitecture] = useState(true)
    const [showSettings, setShowSettings] = useState(false)

    const fileInputRef = useRef(null)
    const messagesEndRef = useRef(null)

    // Check auth on mount
    useEffect(() => {
        const checkAuth = async () => {
            if (api.getToken()) {
                try {
                    const userData = await api.getCurrentUser()
                    setUser(userData)
                } catch (err) {
                    console.error('Auth verification failed:', err)
                    api.logout()
                }
            }
            setAuthChecked(true)
        }
        checkAuth()
    }, [])

    // Auto scroll to bottom of messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    // Handle multiple file selection
    const handleFileSelect = (e) => {
        const selectedFiles = Array.from(e.target.files)
        if (selectedFiles.length > 0) {
            setFiles(prev => [...prev, ...selectedFiles])
            // Do not clear processedDocs to allow adding more
            setMessages([])
            setExtraction(null)
            setPipelineStage(null)
        }
    }

    // Handle drag and drop multiple files
    const handleDrop = (e) => {
        e.preventDefault()
        const droppedFiles = Array.from(e.dataTransfer.files)
        if (droppedFiles.length > 0) {
            setFiles(prev => [...prev, ...droppedFiles])
            setMessages([])
            setExtraction(null)
            setPipelineStage(null)
        }
    }

    const handleDragOver = (e) => {
        e.preventDefault()
    }

    // Remove pending file
    const handleRemoveFile = (index) => {
        setFiles(prev => prev.filter((_, i) => i !== index))
    }

    // Delete processed document
    const handleDeleteDocument = async (documentId) => {
        try {
            await api.request(`/documents/${documentId}`, {
                method: 'DELETE'
            })
            setProcessedDocs(prev => prev.filter(d => d.document_id !== documentId))

            // Clear extraction if deleted doc was target
            if (extraction && extraction.document_id === documentId) {
                setExtraction(null)
            }

            setMessages(prev => [...prev, {
                type: 'assistant',
                content: 'Document removed.',
                timestamp: new Date()
            }])

        } catch (error) {
            console.error('Error deleting document:', error)
        }
    }

    // Parallel Upload with pipeline visualization
    const handleUpload = async () => {
        if (files.length === 0) return

        setUploading(true)
        setShowArchitecture(false)
        setPipelineStage('parsing')

        try {
            // Simulate pipeline startup
            await new Promise(r => setTimeout(r, 600))
            setPipelineStage('chunking')

            // Parallel Uploads using api.request
            const uploadPromises = files.map(async (file) => {
                const formData = new FormData()
                formData.append('file', file)

                return api.request('/upload', {
                    method: 'POST',
                    body: formData,
                })
            })

            // Wait for all to complete concurrently
            setPipelineStage('embedding')
            const results = await Promise.all(uploadPromises)

            setPipelineStage('indexing')
            await new Promise(r => setTimeout(r, 600))

            setPipelineStage('complete')
            await new Promise(r => setTimeout(r, 1000))
            setPipelineStage(null)

            // Add to processed list
            setProcessedDocs(prev => [...prev, ...results])
            setFiles([]) // Clear queue

            setMessages([{
                type: 'assistant',
                content: `Success! Processed ${results.length} new document(s). Total ${results.reduce((acc, d) => acc + d.chunks_created, 0)} chunks index. You can now ask questions across ALL ${processedDocs.length + results.length} documents.`,
                timestamp: new Date()
            }])
        } catch (error) {
            setPipelineStage('error')
            setMessages([{
                type: 'assistant',
                content: `Error uploading documents: ${error.message}`,
                isError: true,
                timestamp: new Date()
            }])
        } finally {
            setUploading(false)
        }
    }

    // Ask question with pipeline visualization
    const handleAsk = async () => {
        if (!question.trim() || processedDocs.length === 0 || asking) return

        const userMessage = {
            type: 'user',
            content: question,
            timestamp: new Date()
        }
        setMessages(prev => [...prev, userMessage])
        const currentQuestion = question
        setQuestion('')
        setAsking(true)
        setShowArchitecture(false)
        setPipelineStage('query-embedding')

        try {
            await new Promise(r => setTimeout(r, 600))
            setPipelineStage('retrieving')
            await new Promise(r => setTimeout(r, 600))
            setPipelineStage('generating')

            // Collect all document IDs
            const docIds = processedDocs.map(d => d.document_id)

            const data = await api.request('/ask', {
                method: 'POST',
                body: JSON.stringify({
                    document_ids: docIds,
                    question: currentQuestion
                })
            })

            setPipelineStage('complete')
            await new Promise(r => setTimeout(r, 1000))
            setPipelineStage(null)

            setMessages(prev => [...prev, {
                type: 'assistant',
                content: data.answer,
                confidence: data.confidence,
                confidenceLevel: data.confidence_level,
                sources: data.sources,
                guardrailStatus: data.guardrail_status,
                guardrailMessage: data.guardrail_message,
                timestamp: new Date()
            }])
        } catch (error) {
            setPipelineStage('error')
            setMessages(prev => [...prev, {
                type: 'assistant',
                content: `Error: ${error.message}`,
                isError: true,
                timestamp: new Date()
            }])
        } finally {
            setAsking(false)
        }
    }

    // Handle Enter key
    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleAsk()
        }
    }

    // Extract structured data with pipeline visualization (Last document only for now)
    const handleExtract = async () => {
        if (processedDocs.length === 0 || extracting) return

        const targetDoc = processedDocs[processedDocs.length - 1]
        setExtracting(true)
        setShowArchitecture(false)
        setPipelineStage('analyzing')

        try {
            await new Promise(r => setTimeout(r, 700))
            setPipelineStage('extracting-fields')
            await new Promise(r => setTimeout(r, 700))
            setPipelineStage('validating')

            const data = await api.request('/extract', {
                method: 'POST',
                body: JSON.stringify({ document_id: targetDoc.document_id })
            })

            setPipelineStage('complete')
            await new Promise(r => setTimeout(r, 1000))
            setPipelineStage(null)
            setExtraction(data)
        } catch (error) {
            setPipelineStage('error')
            setExtraction({ error: error.message })
        } finally {
            setExtracting(false)
        }
    }

    // Log Out
    const handleLogout = () => {
        api.logout()
        setUser(null)
        window.location.reload()
    }

    // Format file size
    const formatFileSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B'
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
    }

    // Get file type icon
    const getFileIcon = (filename) => {
        const ext = filename?.split('.').pop()?.toLowerCase()
        if (ext === 'pdf') return '📄'
        if (ext === 'docx' || ext === 'doc') return '📝'
        return '📃'
    }

    if (!authChecked) return null
    if (!user) return <Login onLoginSuccess={setUser} />

    return (
        <div className="app">
            {showSettings && (
                <SettingsModal
                    onClose={() => setShowSettings(false)}
                />
            )}

            <header className="header" style={{ position: 'relative' }}>
                <button
                    className="btn-settings"
                    onClick={() => setShowSettings(true)}
                    style={{
                        position: 'absolute',
                        top: 0,
                        right: 0,
                        background: 'transparent',
                        border: '1px solid var(--border)',
                        color: 'var(--text-secondary)',
                        padding: '8px 16px',
                        borderRadius: 'var(--radius-sm)',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        fontSize: '0.9rem',
                        transition: 'all 0.2s'
                    }}
                >
                    ⚙️ Settings
                </button>
                <button
                    className="btn-logout"
                    onClick={handleLogout}
                    style={{
                        position: 'absolute',
                        top: 0,
                        right: '110px',
                        background: 'transparent',
                        border: '1px solid rgba(239, 68, 68, 0.3)',
                        color: '#fca5a5',
                        padding: '8px 16px',
                        borderRadius: 'var(--radius-sm)',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        fontSize: '0.9rem',
                        transition: 'all 0.2s'
                    }}
                >
                    🔒 Logout
                </button>
                <h1>🔮 <span className="gradient">Ultra Doc-Intelligence</span></h1>
                <p>AI-powered logistics document assistant with RAG, guardrails & structured extraction</p>
            </header>

            {/* Animated Pipeline Diagram - Shows during processing */}
            {pipelineStage && <AnimatedPipelineDiagram stage={pipelineStage} />}

            {/* Static Architecture Diagram - Shows when idle */}
            {showArchitecture && !pipelineStage && <ArchitectureDiagram />}

            <div className="main-layout">
                {/* Left Panel - Upload & Document Info */}
                <div className="left-panel">
                    <div className="card">
                        <div className="card-header">
                            <div className="card-icon upload">📤</div>
                            <h2>Document Upload</h2>
                        </div>

                        <div
                            className="upload-area"
                            onClick={() => fileInputRef.current?.click()}
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                multiple
                                accept=".pdf,.docx,.txt"
                                onChange={handleFileSelect}
                                style={{ display: 'none' }}
                            />

                            <div className="upload-icon">📁</div>
                            <p className="upload-text">
                                <strong>Click to add files</strong> or drag and drop<br />
                                Upload multiple PDF, DOCX, TXT
                            </p>
                        </div>

                        {/* File List */}
                        {(files.length > 0 || processedDocs.length > 0) && (
                            <div className="file-list" style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                {/* Processed Docs */}
                                {processedDocs.map(doc => (
                                    <div key={doc.document_id} className="file-item" style={{ display: 'flex', alignItems: 'center', padding: '8px', background: 'rgba(52, 211, 153, 0.1)', borderRadius: '6px', border: '1px solid rgba(52, 211, 153, 0.3)' }}>
                                        <span style={{ marginRight: '8px' }}>✅</span>
                                        <span className="file-name" style={{ fontSize: '0.9rem', flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{doc.filename}</span>
                                        <span style={{ fontSize: '0.75rem', opacity: 0.7, marginRight: '8px' }}>{doc.chunks_created} chunks</span>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); handleDeleteDocument(doc.document_id); }}
                                            style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: '0 4px', fontSize: '1.2rem', lineHeight: 1 }}
                                            title="Remove document"
                                        >
                                            ×
                                        </button>
                                    </div>
                                ))}

                                {/* Pending Files */}
                                {files.map((f, i) => (
                                    <div key={i} className="file-item" style={{ display: 'flex', alignItems: 'center', padding: '8px', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '6px' }}>
                                        <span style={{ marginRight: '8px' }}>📄</span>
                                        <span className="file-name" style={{ fontSize: '0.9rem', flex: 1 }}>{f.name}</span>
                                        <span style={{ fontSize: '0.75rem', opacity: 0.7, marginRight: '8px' }}>{(f.size / 1024).toFixed(0)} KB</span>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); handleRemoveFile(i); }}
                                            style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', padding: '0 4px', fontSize: '1.2rem', lineHeight: 1 }}
                                            title="Remove file"
                                        >
                                            ×
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}

                        <button
                            className="btn btn-primary"
                            onClick={handleUpload}
                            disabled={files.length === 0 || uploading}
                            style={{ marginTop: '1rem' }}
                        >
                            {uploading ? (
                                <>
                                    <span className="spinner"></span>
                                    Processing {files.length} files...
                                </>
                            ) : (
                                '🚀 Process All Files'
                            )}
                        </button>

                        {processedDocs.length > 0 && (
                            <div className="document-status">
                                <div className="status-header">
                                    <span>✓</span>
                                    {processedDocs.length} Documents Ready
                                </div>
                                <div className="status-details">
                                    You can now ask questions across all uploaded documents.
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Panel - Chat & Extraction */}
                <div className="right-panel">
                    <div className="card">
                        <div className="tabs">
                            <button
                                className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
                                onClick={() => setActiveTab('chat')}
                            >
                                💬 Ask Questions
                            </button>
                            <button
                                className={`tab ${activeTab === 'extract' ? 'active' : ''}`}
                                onClick={() => setActiveTab('extract')}
                            >
                                📊 Extract Data
                            </button>
                        </div>

                        {activeTab === 'chat' ? (
                            <div className="chat-container">
                                <div className="chat-messages">
                                    {messages.length === 0 ? (
                                        <div className="empty-state">
                                            <div className="icon">💬</div>
                                            <h3>No messages yet</h3>
                                            <p>Upload a document and start asking questions</p>
                                        </div>
                                    ) : (
                                        messages.map((msg, idx) => (
                                            <Message key={idx} message={msg} />
                                        ))
                                    )}
                                    <div ref={messagesEndRef} />
                                </div>

                                <div className="chat-input-container">
                                    <input
                                        type="text"
                                        className="chat-input"
                                        placeholder={processedDocs.length > 0 ? "Ask a question about your documents..." : "Upload documents first..."}
                                        value={question}
                                        onChange={(e) => setQuestion(e.target.value)}
                                        onKeyPress={handleKeyPress}
                                        disabled={processedDocs.length === 0 || asking}
                                    />
                                    <button
                                        className="btn-send"
                                        onClick={handleAsk}
                                        disabled={processedDocs.length === 0 || !question.trim() || asking}
                                    >
                                        {asking ? <span className="spinner"></span> : '➤'}
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <div className="extraction-panel">
                                <p>
                                    Extract structured shipment data from your logistics document into JSON format.
                                </p>

                                <button
                                    className="btn btn-primary"
                                    onClick={handleExtract}
                                    disabled={processedDocs.length === 0 || extracting}
                                    style={{ width: 'auto', marginTop: 0 }}
                                >
                                    {extracting ? (
                                        <>
                                            <span className="spinner"></span>
                                            Extracting...
                                        </>
                                    ) : (
                                        '🔍 Run Extraction'
                                    )}
                                </button>

                                {extraction && !extraction.error && (
                                    <>
                                        <div style={{ marginTop: '24px', display: 'flex', gap: '16px', alignItems: 'center', flexWrap: 'wrap' }}>
                                            <span className={`confidence-badge ${extraction.extraction_confidence >= 0.8 ? 'high' : extraction.extraction_confidence >= 0.5 ? 'medium' : 'low'}`}>
                                                {extraction.extraction_confidence >= 0.8 ? '✓' : extraction.extraction_confidence >= 0.5 ? '◐' : '⚠'}
                                                {' '}Confidence: {(extraction.extraction_confidence * 100).toFixed(0)}%
                                            </span>
                                            <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                                                {extraction.fields_found} of {extraction.total_fields} fields extracted
                                            </span>
                                        </div>

                                        <div className="extraction-grid">
                                            {Object.entries(extraction.extraction).map(([key, value]) => (
                                                <div key={key} className="extraction-field">
                                                    <label>{key.replace(/_/g, ' ')}</label>
                                                    <div className={`value ${value === null ? 'missing' : ''}`}>
                                                        {value !== null ? String(value) : 'Not found'}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </>
                                )}

                                {extraction?.error && (
                                    <div className="guardrail-alert error" style={{ marginTop: '20px' }}>
                                        <span>⚠️</span>
                                        <span>{extraction.error}</span>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

// Static Architecture Diagram Component
function ArchitectureDiagram() {
    return (
        <div className="architecture-diagram">
            <div className="arch-title">🏗️ RAG Architecture Pipeline</div>
            <div className="arch-flow">
                <div className="arch-node doc">
                    <div className="arch-icon">📄</div>
                    <div className="arch-label">Document</div>
                </div>
                <div className="arch-arrow">→</div>
                <div className="arch-node parse">
                    <div className="arch-icon">📋</div>
                    <div className="arch-label">Parse</div>
                </div>
                <div className="arch-arrow">→</div>
                <div className="arch-node chunk">
                    <div className="arch-icon">✂️</div>
                    <div className="arch-label">Chunk</div>
                </div>
                <div className="arch-arrow">→</div>
                <div className="arch-node embed">
                    <div className="arch-icon">🧮</div>
                    <div className="arch-label">Embed</div>
                </div>
                <div className="arch-arrow">→</div>
                <div className="arch-node vector">
                    <div className="arch-icon">📊</div>
                    <div className="arch-label">Store</div>
                </div>
                <div className="arch-arrow">→</div>
                <div className="arch-node search">
                    <div className="arch-icon">🔍</div>
                    <div className="arch-label">Search</div>
                </div>
                <div className="arch-arrow">→</div>
                <div className="arch-node llm">
                    <div className="arch-icon">🤖</div>
                    <div className="arch-label">LLM</div>
                </div>
                <div className="arch-arrow">→</div>
                <div className="arch-node answer">
                    <div className="arch-icon">💬</div>
                    <div className="arch-label">Answer</div>
                </div>
            </div>
            <div className="arch-subtitle">Upload a document to see the pipeline in action!</div>
        </div>
    )
}

// Animated Pipeline Diagram Component
function AnimatedPipelineDiagram({ stage }) {
    const stages = [
        { id: 'parsing', icon: '📄', label: 'Parsing Document', desc: 'Extracting text from file' },
        { id: 'chunking', icon: '✂️', label: 'Chunking', desc: 'Splitting into segments' },
        { id: 'embedding', icon: '🧮', label: 'Embedding', desc: 'Creating vectors' },
        { id: 'indexing', icon: '📊', label: 'Indexing', desc: 'Storing in ChromaDB' },
        { id: 'query-embedding', icon: '🔤', label: 'Encoding Query', desc: 'Converting question to vector' },
        { id: 'retrieving', icon: '🔍', label: 'Retrieving', desc: 'Finding similar chunks' },
        { id: 'generating', icon: '🤖', label: 'Generating', desc: 'LLM processing with Gemini' },
        { id: 'analyzing', icon: '📋', label: 'Analyzing', desc: 'Reading full document' },
        { id: 'extracting-fields', icon: '🎯', label: 'Extracting', desc: 'Pulling structured fields' },
        { id: 'validating', icon: '✅', label: 'Validating', desc: 'Checking data format' },
        { id: 'complete', icon: '✨', label: 'Complete!', desc: 'Processing finished' },
        { id: 'error', icon: '❌', label: 'Error', desc: 'Something went wrong' }
    ]

    const currentStage = stages.find(s => s.id === stage) || stages[0]
    const currentIndex = stages.findIndex(s => s.id === stage)

    // Determine which stages to show based on context
    let visibleStages = []
    if (['parsing', 'chunking', 'embedding', 'indexing'].includes(stage)) {
        visibleStages = stages.slice(0, 4)
    } else if (['query-embedding', 'retrieving', 'generating'].includes(stage)) {
        visibleStages = stages.slice(4, 7)
    } else if (['analyzing', 'extracting-fields', 'validating'].includes(stage)) {
        visibleStages = stages.slice(7, 10)
    } else {
        visibleStages = [currentStage]
    }

    return (
        <div className="pipeline-diagram">
            <div className="pipeline-header">
                <span className="pipeline-icon-large">{currentStage.icon}</span>
                <div className="pipeline-info">
                    <div className="pipeline-label-large">{currentStage.label}</div>
                    <div className="pipeline-desc">{currentStage.desc}</div>
                </div>
            </div>

            <div className="pipeline-stages">
                {visibleStages.map((s, idx) => {
                    const isActive = s.id === stage
                    const isPast = visibleStages.findIndex(vs => vs.id === stage) > idx
                    return (
                        <div key={s.id} className={`pipeline-stage ${isActive ? 'active' : ''} ${isPast ? 'past' : ''}`}>
                            <div className="stage-icon">{s.icon}</div>
                            <div className="stage-label">{s.label.split(' ')[0]}</div>
                            {idx < visibleStages.length - 1 && (
                                <div className={`stage-connector ${isPast || isActive ? 'filled' : ''}`}>
                                    <div className="connector-line"></div>
                                    <div className="connector-arrow">▶</div>
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>

            <div className="pipeline-progress-bar">
                <div
                    className="pipeline-progress-fill"
                    style={{ width: `${((visibleStages.findIndex(s => s.id === stage) + 1) / visibleStages.length) * 100}%` }}
                ></div>
            </div>
        </div>
    )
}

// Message Component
function Message({ message }) {
    const [showSources, setShowSources] = useState(false)

    if (message.type === 'user') {
        return (
            <div className="message message-user">
                <div className="message-bubble">{message.content}</div>
            </div>
        )
    }

    return (
        <div className="message message-assistant">
            <div className="message-bubble">
                <div className="message-content">{message.content}</div>

                {message.confidence !== undefined && (
                    <div className="confidence-stats">
                        <span className={`confidence-badge ${message.confidenceLevel}`}>
                            {message.confidenceLevel === 'high' ? '✓' : message.confidenceLevel === 'medium' ? '◐' : '⚠'}
                            {' '}Confidence: {(message.confidence * 100).toFixed(0)}%
                        </span>

                        {message.guardrailStatus !== 'passed' && message.guardrailMessage && (
                            <span style={{ fontSize: '0.8rem', color: 'var(--warning)' }}>
                                ⚠️ {message.guardrailStatus}
                            </span>
                        )}
                    </div>
                )}

                {message.sources && message.sources.length > 0 && (
                    <div className="sources-section">
                        <button
                            className="sources-toggle"
                            onClick={() => setShowSources(!showSources)}
                        >
                            {showSources ? '▼' : '▶'} View {message.sources.length} source{message.sources.length > 1 ? 's' : ''}
                        </button>

                        {showSources && (
                            <div className="sources-list">
                                {message.sources.map((source, idx) => (
                                    <div key={idx} className="source-item">
                                        <div className="source-header">
                                            <span>📄 Source {idx + 1}</span>
                                            {source.page && <span>• Page {source.page}</span>}
                                            <span>• {(source.similarity_score * 100).toFixed(0)}% match</span>
                                        </div>
                                        <div className="source-text">{source.text}</div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    )
}

export default App
