"""Audio validation and a lazy, serialized F5-TTS integration."""
import hashlib
import json
import math
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf


def inspect_audio(path):
    if not path or not Path(path).is_file():
        raise ValueError("Upload a reference WAV or FLAC recording first.")
    if Path(path).stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Reference audio must be smaller than 20 MB.")
    try:
        info = sf.info(path)
        if not 3 <= info.duration <= 12:
            raise ValueError("Choose a complete 3–12 second excerpt; audio is never silently trimmed.")
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError("Cannot decode this recording. Export a WAV or FLAC file.") from exc
    if not np.isfinite(audio).all():
        raise ValueError("Recording contains invalid samples.")
    mono = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    if rms < 0.001:
        raise ValueError("Recording is silent or too quiet. Record closer to the microphone.")
    if sr < 16000:
        raise ValueError("Use a recording sampled at 16 kHz or higher.")
    warnings = []
    if float(np.mean(np.abs(audio) >= 0.999)) > 0.001:
        warnings.append("Possible clipping: lower the microphone gain and record again.")
    if rms < 0.01:
        warnings.append("Low recording level may reduce quality.")
    return mono, sr, {"duration_seconds": round(len(mono) / sr, 3),
                      "sample_rate": sr, "channels": info.channels,
                      "rms": round(rms, 5), "warnings": warnings}


class VoiceService:
    def __init__(self, output_dir="data/runs", model_factory=None):
        self.output_dir = Path(output_dir)
        self.model_factory = model_factory
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is None:
            if self.model_factory:
                self._model = self.model_factory()
            else:
                if not shutil.which("ffmpeg"):
                    raise RuntimeError("FFmpeg is missing. Install it, then restart VoiceMimic. See README.")
                try:
                    from f5_tts.api import F5TTS
                except ImportError as exc:
                    raise RuntimeError('Voice engine is missing. Run: pip install -e ".[voice]"') from exc
                self._model = F5TTS(model="F5TTS_v1_Base")
        return self._model

    def generate(self, reference, transcript, text, consent, speed=1.0, seed=42):
        if consent is not True:
            raise ValueError("Confirm that this is your voice or you have the speaker's permission.")
        transcript, text = (transcript or "").strip(), (text or "").strip()
        if not transcript or len(transcript) > 1500:
            raise ValueError("Enter the exact reference transcript (1–1,500 characters).")
        if not text or len(text) > 600:
            raise ValueError("Enter 1–600 characters of new speech.")
        if not math.isfinite(float(speed)) or not 0.7 <= float(speed) <= 1.3:
            raise ValueError("Speed must be between 0.7 and 1.3.")
        if not math.isfinite(float(seed)) or float(seed) != int(seed) or not 0 <= int(seed) <= 2147483647:
            raise ValueError("Seed must be a whole number between 0 and 2147483647.")
        mono, sr, quality = inspect_audio(reference)
        # F5 sets global random seeds, so model load and inference share one lock.
        with self._lock:
            model = self._load_model()
            run = self.output_dir / uuid.uuid4().hex
            run.mkdir(parents=True)
            try:
                normalized = run / "reference.wav"
                generated = run / "generated-ai.wav"
                sf.write(normalized, mono, sr)
                started = time.monotonic()
                wav, sample_rate, _ = model.infer(
                    ref_file=str(normalized), ref_text=transcript, gen_text=text,
                    speed=float(speed), seed=int(seed), nfe_step=32,
                )
                wav = np.asarray(wav)
                if wav.size == 0 or not np.isfinite(wav).all() or np.max(np.abs(wav)) < 1e-6:
                    raise RuntimeError("The model returned empty or invalid audio. Try another reference clip.")
                sf.write(generated, wav, sample_rate)
                metadata = {"ai_generated": True, "model": "F5TTS_v1_Base",
                            "reference_transcript": transcript, "generated_text": text,
                            "reference_sha256": hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
                            "speaker_permission_confirmed": True, "seed": int(seed),
                            "speed": float(speed), "reference_quality": quality,
                            "generation_seconds": round(time.monotonic() - started, 2),
                            "output_sample_rate": sample_rate}
                manifest = run / "run.json"
                manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                return str(generated), str(manifest), metadata
            except Exception:
                shutil.rmtree(run, ignore_errors=True)
                raise
