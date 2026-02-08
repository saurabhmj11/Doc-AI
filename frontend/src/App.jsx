import { useState, useRef, useEffect } from 'react'

function App() {
    // State management
    const [file, setFile] = useState(null)
    const [documentId, setDocumentId] = useState(null)
    const [documentInfo, setDocumentInfo] = useState(null)
    const [uploading, setUploading] = useState(false)
    const [messages, setMessages] = useState([])
    const [question, setQuestion] = useState('')
    const [asking, setAsking] = useState(false)
    const [activeTab, setActiveTab] = useState('chat')
    const [extraction, setExtraction] = useState(null)
    const [extracting, setExtracting] = useState(false)
    const [pipelineStage, setPipelineStage] = useState(null)
    const [showArchitecture, setShowArchitecture] = useState(true)

    const fileInputRef = useRef(null)
    const messagesEndRef = useRef(null)

    // Auto scroll to bottom of messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    // Handle file selection
    const handleFileSelect = (e) => {
        const selectedFile = e.target.files[0]
        if (selectedFile) {
            setFile(selectedFile)
            setDocumentId(null)
            setDocumentInfo(null)
            setMessages([])
            setExtraction(null)
            setPipelineStage(null)
        }
    }

    // Handle drag and drop
    const handleDrop = (e) => {
        e.preventDefault()
        const droppedFile = e.dataTransfer.files[0]
        if (droppedFile) {
            setFile(droppedFile)
            setDocumentId(null)
            setDocumentInfo(null)
            setMessages([])
            setExtraction(null)
            setPipelineStage(null)
        }
    }

    const handleDragOver = (e) => {
        e.preventDefault()
    }

    // Upload document with pipeline visualization
    const handleUpload = async () => {
        if (!file) return

        setUploading(true)
        setShowArchitecture(false)
        setPipelineStage('parsing')

        const formData = new FormData()
        formData.append('file', file)

        try {
            // Simulate pipeline stages for visualization
            await new Promise(r => setTimeout(r, 800))
            setPipelineStage('chunking')
            await new Promise(r => setTimeout(r, 800))
            setPipelineStage('embedding')
            await new Promise(r => setTimeout(r, 600))
            setPipelineStage('indexing')

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || 'Upload failed')
            }

            const data = await response.json()
            setPipelineStage('complete')
            await new Promise(r => setTimeout(r, 1500))
            setPipelineStage(null)
            setDocumentId(data.document_id)
            setDocumentInfo(data)
            setMessages([{
                type: 'assistant',
                content: `Document "${data.filename}" processed successfully! Created ${data.chunks_created} searchable chunks. You can now ask questions about your document.`,
                timestamp: new Date()
            }])
        } catch (error) {
            setPipelineStage('error')
            setMessages([{
                type: 'assistant',
                content: `Error uploading document: ${error.message}`,
                isError: true,
                timestamp: new Date()
            }])
        } finally {
            setUploading(false)
        }
    }

    // Ask question with pipeline visualization
    const handleAsk = async () => {
        if (!question.trim() || !documentId || asking) return

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

            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    document_id: documentId,
                    question: currentQuestion
                })
            })

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || 'Failed to get answer')
            }

            const data = await response.json()
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

    // Extract structured data with pipeline visualization
    const handleExtract = async () => {
        if (!documentId || extracting) return

        setExtracting(true)
        setShowArchitecture(false)
        setPipelineStage('analyzing')

        try {
            await new Promise(r => setTimeout(r, 700))
            setPipelineStage('extracting-fields')
            await new Promise(r => setTimeout(r, 700))
            setPipelineStage('validating')

            const response = await fetch('/api/extract', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ document_id: documentId })
            })

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || 'Extraction failed')
            }

            const data = await response.json()
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

    return (
        <div className="app">
            <header className="header">
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
                            className={`upload-area ${file ? 'has-file' : ''}`}
                            onClick={() => fileInputRef.current?.click()}
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".pdf,.docx,.txt"
                                onChange={handleFileSelect}
                                style={{ display: 'none' }}
                            />

                            {file ? (
                                <>
                                    <div className="upload-icon">{getFileIcon(file.name)}</div>
                                    <div className="file-info">
                                        <span className="file-icon">{getFileIcon(file.name)}</span>
                                        <div className="file-details">
                                            <div className="file-name">{file.name}</div>
                                            <div className="file-meta">{formatFileSize(file.size)}</div>
                                        </div>
                                    </div>
                                </>
                            ) : (
                                <>
                                    <div className="upload-icon">📁</div>
                                    <p className="upload-text">
                                        <strong>Click to upload</strong> or drag and drop<br />
                                        PDF, DOCX, or TXT (max 10MB)
                                    </p>
                                </>
                            )}
                        </div>

                        <button
                            className="btn btn-primary"
                            onClick={handleUpload}
                            disabled={!file || uploading || documentId}
                        >
                            {uploading ? (
                                <>
                                    <span className="spinner"></span>
                                    Processing...
                                </>
                            ) : documentId ? (
                                '✓ Document Ready'
                            ) : (
                                '🚀 Upload & Process'
                            )}
                        </button>

                        {documentInfo && (
                            <div className="document-status">
                                <div className="status-header">
                                    <span>✓</span>
                                    Document Processed
                                </div>
                                <div className="status-details">
                                    <strong>{documentInfo.chunks_created}</strong> chunks created<br />
                                    ID: <code style={{ fontSize: '0.75rem', opacity: 0.7 }}>{documentInfo.document_id.slice(0, 12)}...</code>
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
                                        placeholder={documentId ? "Ask a question about your document..." : "Upload a document first..."}
                                        value={question}
                                        onChange={(e) => setQuestion(e.target.value)}
                                        onKeyPress={handleKeyPress}
                                        disabled={!documentId || asking}
                                    />
                                    <button
                                        className="btn-send"
                                        onClick={handleAsk}
                                        disabled={!documentId || !question.trim() || asking}
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
                                    disabled={!documentId || extracting}
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
