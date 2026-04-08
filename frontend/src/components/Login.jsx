import { useState } from 'react';
import api from '../services/api';

function Login({ onLoginSuccess }) {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            await api.login(username, password);
            const user = await api.getCurrentUser();
            onLoginSuccess(user);
        } catch (err) {
            setError(err.message || 'Invalid credentials');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="login-overlay">
            <div className="login-card glass">
                <div className="login-header">
                    <span className="login-logo">🔮</span>
                    <h2>Welcome Back</h2>
                    <p>Sign in to Ultra Doc-Intelligence</p>
                </div>

                <form onSubmit={handleSubmit} className="login-form">
                    <div className="form-group">
                        <label>Username</label>
                        <input
                            type="text"
                            placeholder="Enter username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                            autoFocus
                        />
                    </div>

                    <div className="form-group">
                        <label>Password</label>
                        <input
                            type="password"
                            placeholder="Enter password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>

                    {error && <div className="login-error">⚠️ {error}</div>}

                    <button 
                        type="submit" 
                        className="btn btn-primary btn-block"
                        disabled={loading}
                    >
                        {loading ? <span className="spinner"></span> : '🔓 Sign In'}
                    </button>
                    
                    <div className="login-footer">
                        <p>Demo Credentials: <code>admin</code> / <code>admin</code></p>
                    </div>
                </form>
            </div>

            <style jsx="true">{`
                .login-overlay {
                    position: fixed;
                    inset: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.15), transparent),
                                radial-gradient(circle at bottom left, rgba(168, 85, 247, 0.15), transparent),
                                #0f172a;
                    z-index: 9999;
                }

                .login-card {
                    width: 100%;
                    max-width: 400px;
                    padding: 2.5rem;
                    border-radius: 1.5rem;
                    animation: slideUp 0.5s ease-out;
                }

                @keyframes slideUp {
                    from { transform: translateY(20px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }

                .login-header {
                    text-align: center;
                    margin-bottom: 2rem;
                }

                .login-logo {
                    font-size: 3rem;
                    display: block;
                    margin-bottom: 1rem;
                }

                .login-header h2 {
                    font-size: 1.75rem;
                    font-weight: 700;
                    margin-bottom: 0.5rem;
                    background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }

                .login-header p {
                    color: var(--text-secondary);
                    font-size: 0.95rem;
                }

                .login-form .form-group {
                    margin-bottom: 1.5rem;
                }

                .login-form label {
                    display: block;
                    font-size: 0.85rem;
                    font-weight: 600;
                    margin-bottom: 0.5rem;
                    color: var(--text-secondary);
                }

                .login-form input {
                    width: 100%;
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    padding: 0.85rem 1rem;
                    border-radius: 0.75rem;
                    color: #fff;
                    font-size: 1rem;
                    transition: all 0.2s;
                }

                .login-form input:focus {
                    outline: none;
                    border-color: rgba(99, 102, 241, 0.5);
                    background: rgba(255, 255, 255, 0.05);
                    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
                }

                .login-error {
                    background: rgba(239, 68, 68, 0.1);
                    border: 1px solid rgba(239, 68, 68, 0.2);
                    color: #fca5a5;
                    padding: 0.75rem;
                    border-radius: 0.75rem;
                    font-size: 0.9rem;
                    margin-bottom: 1.5rem;
                    text-align: center;
                }

                .btn-block {
                    width: 100%;
                    padding: 1rem;
                    font-size: 1rem;
                    font-weight: 600;
                }

                .login-footer {
                    margin-top: 1.5rem;
                    text-align: center;
                    font-size: 0.85rem;
                    color: var(--text-muted);
                }

                .login-footer code {
                    background: rgba(255, 255, 255, 0.1);
                    padding: 2px 6px;
                    border-radius: 4px;
                    color: var(--primary-light);
                }
            `}</style>
        </div>
    );
}

export default Login;
