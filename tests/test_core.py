import json

import numpy as np
import pytest
import soundfile as sf

from voicemimic.core import VoiceService, inspect_audio


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "reference.wav"
    t = np.arange(4 * 24000) / 24000
    sf.write(path, 0.2 * np.sin(2 * np.pi * 220 * t), 24000)
    return str(path)


class FakeModel:
    def infer(self, **kwargs):
        assert kwargs["ref_text"] == "Hello there."
        assert kwargs["gen_text"] == "A new sentence."
        return np.full(24000, 0.1), 24000, None


def test_generation_writes_audio_and_provenance(reference, tmp_path):
    service = VoiceService(tmp_path / "runs", model_factory=FakeModel)
    audio, manifest, metadata = service.generate(reference, "Hello there.", "A new sentence.", True)
    assert sf.info(audio).duration == 1
    assert json.loads(open(manifest).read())["ai_generated"] is True
    assert len(metadata["reference_sha256"]) == 64
    assert metadata["reference_quality"]["duration_seconds"] == 4


@pytest.mark.parametrize("changes", [
    {"consent": False}, {"transcript": ""}, {"text": ""}, {"text": "x" * 601},
    {"speed": 0}, {"speed": float("nan")}, {"seed": -1}, {"seed": 1.5},
])
def test_bad_requests_never_load_model(reference, tmp_path, changes):
    def forbidden():
        pytest.fail("Invalid requests must not load the model")
    args = dict(reference=reference, transcript="Hello there.", text="A new sentence.", consent=True)
    args.update(changes)
    with pytest.raises(ValueError):
        VoiceService(tmp_path, model_factory=forbidden).generate(**args)


@pytest.mark.parametrize("seconds,amplitude", [(1, .2), (13, .2), (4, 0)])
def test_rejects_unusable_recordings(tmp_path, seconds, amplitude):
    path = tmp_path / "bad.wav"
    sf.write(path, np.full(seconds * 24000, amplitude, dtype=np.float32), 24000)
    with pytest.raises(ValueError):
        inspect_audio(str(path))


def test_failed_inference_cleans_partial_run(reference, tmp_path):
    class BrokenModel:
        def infer(self, **kwargs):
            raise RuntimeError("device failure")
    root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="device failure"):
        VoiceService(root, BrokenModel).generate(reference, "Hello", "New text", True)
    assert list(root.iterdir()) == []


def test_model_is_reused(reference, tmp_path):
    calls = []
    def factory():
        calls.append(1)
        return FakeModel()
    service = VoiceService(tmp_path / "runs", factory)
    for _ in range(2):
        service.generate(reference, "Hello there.", "A new sentence.", True)
    assert len(calls) == 1
