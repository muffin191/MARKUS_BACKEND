from flask import Blueprint, request, send_file, jsonify
import base64
import io
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import zipfile

import requests

sts = Blueprint("sts", __name__)

_embed_lock = threading.Lock()
_cached_embed = None

_models_lock = threading.Lock()
_models_loaded = False
_rtvc_checked = False
_rtvc_ready = False
_rtvc_error = ""

encoder = None
Synthesizer = None
vocoder = None
np = None
whisper = None
synthesizer = None
whisper_model = None

RTVC_ROOT = Path(__file__).parent / "Real-Time-Voice-Cloning"
ENCODER_MODEL_PATH = RTVC_ROOT / "saved_models/default/encoder.pt"
SYNTHESIZER_MODEL_PATH = RTVC_ROOT / "saved_models/default/synthesizer.pt"
VOCODER_MODEL_PATH = RTVC_ROOT / "saved_models/default/vocoder.pt"
RTVC_REPO_ZIP_URL = os.getenv(
    "RTVC_REPO_ZIP_URL",
    "https://codeload.github.com/CorentinJ/Real-Time-Voice-Cloning/zip/refs/heads/master",
)


def _download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _bootstrap_rtvc_repo() -> None:
    if RTVC_ROOT.exists():
        return

    logging.info("STS: downloading RTVC repository archive...")
    temp_dir = Path(tempfile.mkdtemp(prefix="rtvc_repo_"))
    archive_path = temp_dir / "rtvc.zip"

    try:
        _download_file(RTVC_REPO_ZIP_URL, archive_path)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        extracted_root = temp_dir / "Real-Time-Voice-Cloning-master"
        if not extracted_root.exists():
            raise RuntimeError("Downloaded RTVC archive does not contain expected root directory")

        shutil.move(str(extracted_root), str(RTVC_ROOT))
        logging.info("STS: RTVC repository extracted to %s", RTVC_ROOT)
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


def _ensure_rtvc_modules() -> bool:
    global _rtvc_checked, _rtvc_ready, _rtvc_error
    global encoder, Synthesizer, vocoder, np, whisper

    if _rtvc_checked:
        return _rtvc_ready

    _rtvc_checked = True
    try:
        _bootstrap_rtvc_repo()
        if not RTVC_ROOT.exists():
            raise FileNotFoundError(f"Real-Time-Voice-Cloning directory not found at {RTVC_ROOT}")

        if str(RTVC_ROOT) not in sys.path:
            sys.path.insert(0, str(RTVC_ROOT))

        import numpy as _np
        import whisper as _whisper
        from encoder import inference as _encoder
        from synthesizer.inference import Synthesizer as _Synthesizer
        from utils.default_models import ensure_default_models as _ensure_default_models
        from vocoder import inference as _vocoder

        encoder = _encoder
        Synthesizer = _Synthesizer
        vocoder = _vocoder
        np = _np
        whisper = _whisper

        # Download default model files from Hugging Face when missing.
        _ensure_default_models(RTVC_ROOT / "saved_models")

        _rtvc_ready = True
        logging.info("STS RTVC stack detected")
    except Exception as e:
        _rtvc_ready = False
        _rtvc_error = str(e)
        logging.warning("STS running in fallback mode: %s", e)

    return _rtvc_ready


def _ensure_models_loaded() -> bool:
    global _models_loaded, synthesizer, whisper_model
    if _models_loaded:
        return True
    if not _ensure_rtvc_modules():
        return False

    with _models_lock:
        if _models_loaded:
            return True
        logging.info("Loading STS RTVC models...")
        encoder.load_model(ENCODER_MODEL_PATH, device="cpu")
        synth = Synthesizer(SYNTHESIZER_MODEL_PATH, verbose=False)
        synth.load()
        vocoder.load_model(VOCODER_MODEL_PATH, verbose=False)
        whisper_model_local = whisper.load_model("base")
        synthesizer = synth
        whisper_model = whisper_model_local
        _models_loaded = True
        logging.info("STS RTVC models loaded")
    return True


def _basic_tts_wav(text: str) -> io.BytesIO:
    """Fallback TTS using pyttsx3 when RTVC stack is unavailable."""
    import pyttsx3

    fd, wav_path = tempfile.mkstemp(prefix="sts_fallback_", suffix=".wav")
    os.close(fd)
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        with open(wav_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass

    out_buffer = io.BytesIO(data)
    out_buffer.seek(0)
    return out_buffer


def _synthesize_with_embed(text, embed):
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, 16000), mode="constant")
    out_buffer = io.BytesIO()
    import soundfile as sf

    sf.write(out_buffer, generated_wav.astype(np.float32), synthesizer.sample_rate, format="WAV")
    out_buffer.seek(0)
    return out_buffer


def _call_chat_backend(message: str, agent_mode: bool = False) -> str:
    chat_url = os.getenv("STS_CHAT_URL", "http://localhost:5000/chat")
    response = requests.post(
        chat_url,
        json={"message": message, "agentMode": agent_mode},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Chat backend returned {response.status_code}")

    payload = response.json()
    reply = (payload.get("reply") or payload.get("response") or "").strip()
    if not reply:
        raise RuntimeError("Chat backend returned an empty reply")
    return reply


@sts.route("/sts/status", methods=["GET"])
def sts_status():
    rtvc_ready = _ensure_rtvc_modules()
    return jsonify(
        {
            "status": "ok",
            "mode": "rtvc" if rtvc_ready else "fallback",
            "rtvc_ready": rtvc_ready,
            "models_loaded": _models_loaded,
            "reason": _rtvc_error if not rtvc_ready else "",
        }
    )


@sts.route("/sts", methods=["POST"])
def speech_to_speech():
    if _ensure_models_loaded():
        if "voice_sample" not in request.files or "source_audio" not in request.files:
            return jsonify({"error": "Please upload both voice_sample and source_audio files."}), 400

        voice_fd, voice_path = tempfile.mkstemp(suffix=".wav")
        source_fd, source_path = tempfile.mkstemp(suffix=".wav")
        os.close(voice_fd)
        os.close(source_fd)
        try:
            request.files["voice_sample"].save(voice_path)
            request.files["source_audio"].save(source_path)
            preprocessed_wav = encoder.preprocess_wav(voice_path)
            embed = encoder.embed_utterance(preprocessed_wav)
            result = whisper_model.transcribe(source_path)
            text = result["text"]
            out_buffer = _synthesize_with_embed(text, embed)
            return send_file(
                out_buffer,
                mimetype="audio/wav",
                as_attachment=True,
                download_name="output.wav",
            )
        finally:
            os.remove(voice_path)
            os.remove(source_path)

    data_text = (request.form.get("text") or request.args.get("text") or "").strip()
    if not data_text:
        return jsonify({"error": "STS fallback mode requires a text field."}), 400

    out_buffer = _basic_tts_wav(data_text)
    return send_file(out_buffer, mimetype="audio/wav", as_attachment=False, download_name="output.wav")


@sts.route("/sts/voice", methods=["POST"])
def set_sts_voice():
    if "voice_sample" not in request.files:
        return jsonify({"error": "Please upload a voice_sample file."}), 400

    voice_fd, voice_path = tempfile.mkstemp(suffix=".wav")
    os.close(voice_fd)
    try:
        request.files["voice_sample"].save(voice_path)
        with open(voice_path, "rb") as f:
            voice_bytes = f.read()

        with _embed_lock:
            global _cached_embed
            if _ensure_models_loaded():
                preprocessed_wav = encoder.preprocess_wav(voice_path)
                _cached_embed = encoder.embed_utterance(preprocessed_wav)
            else:
                _cached_embed = voice_bytes
        return jsonify({"status": "ok"}), 200
    finally:
        os.remove(voice_path)


@sts.route("/sts/speak", methods=["POST"])
def speak_with_cached_voice():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    with _embed_lock:
        embed = _cached_embed

    if _ensure_models_loaded() and embed is not None and not isinstance(embed, (bytes, bytearray)):
        out_buffer = _synthesize_with_embed(text, embed)
    else:
        out_buffer = _basic_tts_wav(text)

    return send_file(out_buffer, mimetype="audio/wav", as_attachment=False, download_name="cloned.wav")


@sts.route("/sts/chat", methods=["POST"])
def sts_chat():
    agent_mode = request.form.get("agentMode", "false").lower() in ("1", "true", "yes", "on")

    transcript = ""
    if _ensure_models_loaded() and "source_audio" in request.files:
        source_file = request.files["source_audio"]
        suffix = Path(source_file.filename or "").suffix or ".wav"
        source_fd, source_path = tempfile.mkstemp(suffix=suffix)
        os.close(source_fd)
        try:
            source_file.save(source_path)
            result = whisper_model.transcribe(source_path)
            transcript = (result.get("text") or "").strip()
        finally:
            os.remove(source_path)

    if not transcript:
        transcript = (
            request.form.get("text")
            or request.args.get("text")
            or (request.get_json(silent=True) or {}).get("text")
            or ""
        ).strip()

    if not transcript:
        return jsonify({"error": "Provide source_audio (rtvc mode) or text (fallback mode)."}), 400

    try:
        reply_text = _call_chat_backend(transcript, agent_mode=agent_mode)
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    with _embed_lock:
        embed = _cached_embed

    if _ensure_models_loaded() and embed is not None and not isinstance(embed, (bytes, bytearray)):
        out_buffer = _synthesize_with_embed(reply_text, embed)
    else:
        out_buffer = _basic_tts_wav(reply_text)

    audio_b64 = base64.b64encode(out_buffer.getvalue()).decode("ascii")
    return jsonify({"transcript": transcript, "reply": reply_text, "audio": audio_b64}), 200
