"""Portable local voice profiles, independent of the conversational model."""
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from voicemimic.core import inspect_audio


class ProfileStore:
    def __init__(self, root="data/voices"):
        self.root = Path(root)

    def load(self, voice_id):
        if not isinstance(voice_id, str) or not re.fullmatch(r"[a-f0-9]{32}", voice_id):
            raise ValueError("Select a saved voice first.")
        folder = self.root / voice_id
        try:
            profile = json.loads((folder / "profile.json").read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("This voice profile is missing or damaged.") from exc
        if not (folder / "reference.wav").is_file():
            raise ValueError("This profile's reference recording is missing.")
        return {**profile, "reference": str(folder / "reference.wav")}

    def choices(self):
        choices = []
        for path in sorted(self.root.glob("*/profile.json")):
            try:
                profile = self.load(path.parent.name)
                choices.append((profile["name"], profile["id"]))
            except (ValueError, KeyError):
                continue
        return choices

    def save(self, name, reference, transcript, consent, speed=1.0, seed=42):
        name, transcript = (name or "").strip(), (transcript or "").strip()
        if not name or len(name) > 80:
            raise ValueError("Give the voice a name of 1–80 characters.")
        if consent is not True:
            raise ValueError("Confirm the speaker's permission before saving a voice.")
        if not transcript or len(transcript) > 1500:
            raise ValueError("Provide the exact reference transcript (1–1,500 characters).")
        if not 0.7 <= float(speed) <= 1.3 or not 0 <= float(seed) <= 2147483647 or int(seed) != float(seed):
            raise ValueError("Invalid speed or seed.")
        audio, sr, quality = inspect_audio(reference)
        voice_id = uuid.uuid4().hex
        folder = self.root / voice_id
        folder.mkdir(parents=True)
        try:
            sf.write(folder / "reference.wav", audio, sr)
            profile = {"version": 1, "id": voice_id, "name": name,
                       "transcript": transcript, "speed": float(speed), "seed": int(seed),
                       "permission_confirmed": True, "created_at": datetime.now(timezone.utc).isoformat(),
                       "quality": quality}
            (folder / "profile.json").write_text(json.dumps(profile, indent=2))
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return self.load(voice_id)
