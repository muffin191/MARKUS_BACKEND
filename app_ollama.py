"""
JARVIS Backend using Ollama (Local AI)
No external APIs needed - runs 100% locally
"""

from flask import Flask, request, jsonify
import requests
import json
import logging
from datetime import datetime
import os
from typing import Dict, List
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None

try:
    from mem0 import MemoryClient
    HAS_MEM0 = True
except ImportError:
    MemoryClient = None
    HAS_MEM0 = False

load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL_OVERRIDE = os.getenv("GOOGLE_MODEL")
GOOGLE_API_VERSION = os.getenv("GOOGLE_API_VERSION", "v1")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "piper")  # piper, espeak, or similar
MEMORY_FILE = os.getenv("JARVIS_MEMORY_FILE", "jarvis_memory.json")
MEM0_API_KEY = os.getenv("MEM0_API_KEY")

if GOOGLE_API_KEY and genai is not None:
    GEMINI_CLIENT = genai.Client(
        api_key=GOOGLE_API_KEY,
        http_options={"api_version": GOOGLE_API_VERSION},
    )
else:
    GEMINI_CLIENT = None

MEM0_CLIENT = None
if HAS_MEM0:
    try:
        MEM0_CLIENT = MemoryClient()
        logger.info("mem0 client initialized successfully")
    except Exception as e:
        logger.warning("Failed to initialize mem0: %s", e)
        HAS_MEM0 = False

_MODEL_CACHE: str | None = None
_PREFERRED_MODELS = [
    "models/gemini-flash-latest",
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
]

# Conversation memory (persisted to disk)
conversation_history: Dict[str, List[dict]] = {}


def _provider_name() -> str:
    return "gemini" if GEMINI_CLIENT is not None else "ollama"


def _get_content_model() -> str:
    global _MODEL_CACHE
    if _MODEL_CACHE:
        return _MODEL_CACHE
    if GOOGLE_MODEL_OVERRIDE:
        _MODEL_CACHE = GOOGLE_MODEL_OVERRIDE
        return GOOGLE_MODEL_OVERRIDE
    if GEMINI_CLIENT is None:
        return OLLAMA_MODEL
    try:
        available = list(GEMINI_CLIENT.models.list())
        available_names = [model.name for model in available]
        for preferred in _PREFERRED_MODELS:
            if preferred in available_names:
                _MODEL_CACHE = preferred
                return preferred
        if available_names:
            _MODEL_CACHE = available_names[0]
            return available_names[0]
    except Exception as e:
        logger.warning("Gemini model selection failed, using fallback: %s", e)
    return "models/gemini-flash-latest"


def _load_memory() -> None:
    """Load memory from disk if available."""
    global conversation_history
    if not os.path.exists(MEMORY_FILE):
        conversation_history = {}
        return

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            conversation_history = data
        else:
            conversation_history = {}
        logger.info("Loaded memory for %d user(s)", len(conversation_history))
    except Exception as e:
        logger.warning("Failed to load memory file: %s", e)
        conversation_history = {}


def _get_mem0_memories(user_id: str) -> str:
    if not HAS_MEM0 or MEM0_CLIENT is None:
        return ""

    try:
        results = MEM0_CLIENT.get_all(user_id=user_id, filters={"user_id": user_id})
        if not results:
            return ""

        raw_memories = results if isinstance(results, list) else results.get("results", [])
        memory_lines: List[str] = []
        for entry in raw_memories[:10]:
            if isinstance(entry, dict):
                memory_text = entry.get("memory", entry.get("text", ""))
            else:
                memory_text = str(entry)

            cleaned = str(memory_text).strip()
            if cleaned:
                memory_lines.append(cleaned)

        return "\n- ".join(memory_lines)
    except Exception as e:
        logger.warning("Could not fetch mem0 memories for %s: %s", user_id, e)
        return ""


def _save_mem0_memory(user_id: str, user_message: str, assistant_message: str) -> None:
    if not HAS_MEM0 or MEM0_CLIENT is None:
        return

    try:
        MEM0_CLIENT.add(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
            user_id=user_id,
        )
    except Exception as e:
        logger.warning("Could not save mem0 memories for %s: %s", user_id, e)


def _save_memory() -> None:
    """Persist memory to disk atomically."""
    try:
        tmp_path = f"{MEMORY_FILE}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, MEMORY_FILE)
    except Exception as e:
        logger.warning("Failed to save memory file: %s", e)


def _append_message(user_id: str, role: str, content: str) -> None:
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
    )
    _save_memory()


_load_memory()

def get_ollama_response(message: str, user_id: str = "default", use_memory: bool = True) -> str:
    """Get response from the configured AI provider."""
    try:
        history = conversation_history.get(user_id, []) if use_memory else []
        
        # Build context from recent conversation
        context = ""
        if history:
            # Use last 5 exchanges for context
            recent = history[-10:]
            for msg in recent:
                context += f"\n{msg['role']}: {msg['content']}"

        if use_memory:
            memories = _get_mem0_memories(user_id)
            if memories:
                context += f"\n\nWhat you remember about this user:\n- {memories}"
        
        # Create the prompt
        full_message = f"{context}\nuser: {message}\nassistant:"
        if GEMINI_CLIENT is not None:
            response = GEMINI_CLIENT.models.generate_content(
                model=_get_content_model(),
                contents=full_message,
            )
            assistant_message = (response.text or "").strip()
        else:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": full_message,
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
                timeout=120,
            )

            if response.status_code != 200:
                logger.error("Ollama error: %s", response.status_code)
                return "Error: Could not get response from AI model"

            result = response.json()
            assistant_message = result.get("response", "").strip()

        if use_memory:
            _append_message(user_id, "user", message)
            _append_message(user_id, "assistant", assistant_message)
            _save_mem0_memory(user_id, message, assistant_message)

        return assistant_message
    
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure 'ollama serve' is running on localhost:11434, or set GOOGLE_API_KEY for a hosted model."
    except Exception as e:
        logger.error(f"Error getting Ollama response: {e}")
        return f"Error: {str(e)}"

def generate_tts(text: str, voice: str = "onyx") -> bytes:
    """Generate TTS audio bytes as WAV.

    Order of attempts:
    1) pyttsx3 (works on Windows with built-in SAPI voices)
    2) piper
    3) espeak-ng
    """
    try:
        # Preferred on Windows desktop.
        try:
            import tempfile
            import pyttsx3

            fd, wav_path = tempfile.mkstemp(prefix="jarvis_tts_", suffix=".wav")
            os.close(fd)
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 180)
                engine.save_to_file(text, wav_path)
                engine.runAndWait()
                with open(wav_path, "rb") as f:
                    data = f.read()
                if data:
                    return data
            finally:
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
        except Exception:
            pass
        
        # Check if piper is available
        try:
            import subprocess
            # Using piper locally if available
            proc = subprocess.run(
                ["piper", "--model", f"en_US-{voice}-medium"],
                input=text.encode(),
                capture_output=True,
                timeout=30
            )
            if proc.returncode == 0:
                return proc.stdout
        except:
            pass
        
        # Fallback: use espeak if piper not available
        try:
            import subprocess
            proc = subprocess.run(
                ["espeak-ng", "-w", "/dev/stdout"],
                input=text.encode(),
                capture_output=True,
                timeout=10
            )
            if proc.returncode == 0:
                return proc.stdout
        except:
            pass
        
        logger.warning("No TTS engine available locally")
        return None
    
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return b"Error generating audio"

# ============= API Routes =============

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    provider = _provider_name()
    if provider == "gemini":
        provider_healthy = GEMINI_CLIENT is not None
        model_name = _get_content_model()
    else:
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            provider_healthy = response.status_code == 200
        except Exception:
            provider_healthy = False
        model_name = OLLAMA_MODEL

    return jsonify({
        "status": "ok" if provider_healthy else "degraded",
        "provider": provider,
        "ollama_running": provider_healthy if provider == "ollama" else False,
        "model": model_name,
        "has_google_api_key": bool(GOOGLE_API_KEY),
        "gemini_library_loaded": genai is not None,
        "gemini_client_ready": GEMINI_CLIENT is not None,
        "mem0_enabled": HAS_MEM0 and MEM0_CLIENT is not None,
        "has_mem0_api_key": bool(MEM0_API_KEY),
        "mem0_library_loaded": MemoryClient is not None,
    })

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint - compatible with your Flutter app"""
    try:
        data = request.json
        message = data.get("message", "").strip()
        use_memory = data.get("use_memory", True)
        enable_tools = data.get("enable_tools", False)
        user_id = data.get("user_id", "default")
        
        if not message:
            return jsonify({"error": "Empty message"}), 400
        
        # Get response from Ollama
        response_text = get_ollama_response(message, user_id, use_memory)
        
        return jsonify({
            "response": response_text,
            "model": OLLAMA_MODEL,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/tts", methods=["POST"])
def tts():
    """Text-to-speech endpoint - compatible with your Flutter app"""
    try:
        data = request.json
        text = data.get("text", "").strip()
        voice = data.get("voice", "onyx")
        
        if not text:
            return jsonify({"error": "Empty text"}), 400
        
        # Generate TTS
        audio_bytes = generate_tts(text, voice)
        
        if audio_bytes is None:
            return jsonify({"error": "TTS not configured"}), 501
        
        return audio_bytes, 200, {"Content-Type": "audio/wav"}
    
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def get_history():
    """Get conversation history for a user"""
    user_id = request.args.get("user_id", "default")
    return jsonify({
        "user_id": user_id,
        "history": conversation_history.get(user_id, [])
    })

@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Clear conversation history"""
    user_id = request.json.get("user_id", "default")
    if user_id in conversation_history:
        del conversation_history[user_id]
        _save_memory()
    return jsonify({"status": "cleared", "user_id": user_id})

@app.route("/models", methods=["GET"])
def list_models():
    """List available Ollama models"""
    if GEMINI_CLIENT is not None:
        current_model = _get_content_model()
        return jsonify({
            "available_models": [current_model],
            "current_model": current_model,
            "provider": "gemini",
        })

    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return jsonify({
                "available_models": [m["name"] for m in models],
                "current_model": OLLAMA_MODEL,
                "provider": "ollama",
            })
    except:
        pass
    
    return jsonify({
        "available_models": [],
        "current_model": OLLAMA_MODEL,
        "provider": "ollama",
        "status": "Unable to connect to Ollama"
    })

if __name__ == "__main__":
    print(f"Starting JARVIS backend with Ollama...")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Memory file: {MEMORY_FILE}")
    print(f"Flask running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
