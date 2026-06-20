# JARVIS Backend

Flask-based AI chat backend using Ollama for local LLM inference.

## Features
- Local LLM via Ollama (phi3:mini)
- Persistent conversation memory (JSON-based)
- Text-to-speech (pyttsx3)
- REST API endpoints for chat, TTS, history management

## Environment Variables
- `OLLAMA_URL` - Ollama API endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL` - Model name (default: phi3:mini)
- `JARVIS_MEMORY_FILE` - Memory file path (default: jarvis_memory.json)

## Running Locally
```bash
pip install -r backend_requirements.txt
python app_ollama.py
```

Server runs on http://localhost:5000

## API Endpoints
- `POST /chat` - Send message and get response
- `POST /tts` - Generate text-to-speech audio
- `GET /history?user_id=default` - Get conversation history
- `POST /clear-history` - Clear conversation history
- `GET /health` - Health check
- `GET /models` - List available models

## Deployment
For Railway/Render: Uses `Procfile` with gunicorn
