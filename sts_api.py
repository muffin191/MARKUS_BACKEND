from flask import Blueprint, request, send_file, jsonify
import os
import io
import tempfile
import whisper
import numpy as np
import torch
from pathlib import Path
import sys
import threading
import base64
import requests
import logging

# Add Real-Time-Voice-Cloning to path
rtvc_path = Path(__file__).parent / "Real-Time-Voice-Cloning"
if rtvc_path.exists():
    sys.path.insert(0, str(rtvc_path))
else:
    raise ImportError(f"Real-Time-Voice-Cloning directory not found at {rtvc_path}")

# Import Real-Time-Voice-Cloning modules
try:
    from encoder import inference as encoder
except ImportError as e:
    raise ImportError(f"Could not import encoder module. Make sure Real-Time-Voice-Cloning is properly installed. Error: {e}")
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

# Set model paths (update if needed)
ENCODER_MODEL_PATH = Path("Real-Time-Voice-Cloning/saved_models/default/encoder.pt")
SYNTHESIZER_MODEL_PATH = Path("Real-Time-Voice-Cloning/saved_models/default/synthesizer.pt")
VOCODER_MODEL_PATH = Path("Real-Time-Voice-Cloning/saved_models/default/vocoder.pt")

_models_lock = threading.Lock()
_models_loaded = False
synthesizer = None
whisper_model = None

sts = Blueprint('sts', __name__)
_cached_embed = None
_embed_lock = threading.Lock()


def _ensure_models_loaded():
    global _models_loaded, synthesizer, whisper_model
    if _models_loaded:
        return
    with _models_lock:
        if _models_loaded:
            return
        logging.info("Loading STS models (lazy init)...")
        encoder.load_model(ENCODER_MODEL_PATH, device="cpu")
        synth = Synthesizer(SYNTHESIZER_MODEL_PATH, verbose=False)
        synth.load()
        vocoder.load_model(VOCODER_MODEL_PATH, verbose=False)
        whisper_model_local = whisper.load_model("base")
        synthesizer = synth
        whisper_model = whisper_model_local
        _models_loaded = True
        logging.info("STS models loaded")


def _synthesize_with_embed(text, embed):
    _ensure_models_loaded()
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])

    # Convert to 16-bit PCM WAV
    generated_wav = np.pad(generated_wav, (0, 16000), mode="constant")
    out_buffer = io.BytesIO()
    import soundfile as sf
    sf.write(out_buffer, generated_wav.astype(np.float32), synthesizer.sample_rate, format='WAV')
    out_buffer.seek(0)
    return out_buffer

@sts.route('/sts', methods=['POST'])
def speech_to_speech():
    _ensure_models_loaded()
    if 'voice_sample' not in request.files or 'source_audio' not in request.files:
        return jsonify({'error': 'Please upload both voice_sample and source_audio files.'}), 400

    # Save uploaded files to temp using mkstemp for Windows compatibility
    voice_fd, voice_path = tempfile.mkstemp(suffix='.wav')
    source_fd, source_path = tempfile.mkstemp(suffix='.wav')
    os.close(voice_fd)
    os.close(source_fd)
    try:
        request.files['voice_sample'].save(voice_path)
        request.files['source_audio'].save(source_path)

        # Step 1: Extract speaker embedding from voice sample
        preprocessed_wav = encoder.preprocess_wav(voice_path)
        embed = encoder.embed_utterance(preprocessed_wav)

        # Step 2: Transcribe source audio to text
        result = whisper_model.transcribe(source_path)
        text = result['text']

        # Step 3: Synthesize speech in cloned voice
        out_buffer = _synthesize_with_embed(text, embed)

        return send_file(out_buffer, mimetype='audio/wav', as_attachment=True, download_name='output.wav')
    finally:
        os.remove(voice_path)
        os.remove(source_path)


@sts.route('/sts/voice', methods=['POST'])
def set_sts_voice():
    _ensure_models_loaded()
    if 'voice_sample' not in request.files:
        return jsonify({'error': 'Please upload a voice_sample file.'}), 400

    voice_fd, voice_path = tempfile.mkstemp(suffix='.wav')
    os.close(voice_fd)
    try:
        request.files['voice_sample'].save(voice_path)
        preprocessed_wav = encoder.preprocess_wav(voice_path)
        embed = encoder.embed_utterance(preprocessed_wav)
        with _embed_lock:
            global _cached_embed
            _cached_embed = embed
        return jsonify({'status': 'ok'}), 200
    finally:
        os.remove(voice_path)


@sts.route('/sts/speak', methods=['POST'])
def speak_with_cached_voice():
    _ensure_models_loaded()
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    with _embed_lock:
        embed = _cached_embed

    if embed is None:
        return jsonify({'error': 'No voice sample set'}), 400

    out_buffer = _synthesize_with_embed(text, embed)
    return send_file(out_buffer, mimetype='audio/wav', as_attachment=False, download_name='cloned.wav')


@sts.route('/sts/chat', methods=['POST'])
def sts_chat():
    _ensure_models_loaded()
    if 'source_audio' not in request.files:
        return jsonify({'error': 'Please upload a source_audio file.'}), 400

    with _embed_lock:
        embed = _cached_embed

    if embed is None:
        return jsonify({'error': 'No voice sample set'}), 400

    source_file = request.files['source_audio']
    suffix = Path(source_file.filename or '').suffix or '.wav'
    source_fd, source_path = tempfile.mkstemp(suffix=suffix)
    os.close(source_fd)

    try:
        source_file.save(source_path)
        result = whisper_model.transcribe(source_path)
        transcript = (result.get('text') or '').strip()
        if not transcript:
            return jsonify({'error': 'Transcription failed'}), 400

        agent_mode = request.form.get('agentMode', 'false').lower() in ('1', 'true', 'yes', 'on')
        chat_url = os.getenv('STS_CHAT_URL', 'http://localhost:5000/chat')

        chat_response = requests.post(
            chat_url,
            json={'message': transcript, 'agentMode': agent_mode},
            timeout=60
        )
        if chat_response.status_code != 200:
            return jsonify({'error': 'Chat backend error'}), 502

        reply_json = chat_response.json()
        reply_text = (reply_json.get('reply') or '').strip()
        if not reply_text:
            return jsonify({'error': 'Empty reply from chat'}), 502

        out_buffer = _synthesize_with_embed(reply_text, embed)
        audio_b64 = base64.b64encode(out_buffer.getvalue()).decode('ascii')
        return jsonify({'transcript': transcript, 'reply': reply_text, 'audio': audio_b64}), 200
    finally:
        os.remove(source_path)
