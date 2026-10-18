import threading

import numpy as np
import pytest
import soundfile as sf

from voicemimic.profiles import ProfileStore
from voicemimic.chat import Turns, split_phrase, speak_reply, OllamaChat


@pytest.fixture
def profile(tmp_path):
    ref = tmp_path / "voice.wav"
    sf.write(ref, np.sin(np.arange(96000) * .04) * .2, 24000)
    return ProfileStore(tmp_path / "voices").save("Voice", str(ref), "Hello world.", True)


def test_profile_survives_reload(profile, tmp_path):
    store = ProfileStore(tmp_path / "voices")
    assert store.load(profile["id"])["transcript"] == "Hello world."
    assert store.choices() == [("Voice", profile["id"])]
    assert sf.info(store.load(profile["id"])["reference"]).duration == 4


def test_profile_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        ProfileStore(tmp_path).load("../secret")


def test_profile_requires_permission(tmp_path):
    with pytest.raises(ValueError, match="permission"):
        ProfileStore(tmp_path).save("Voice", "unused", "Hello", False)
    assert list(tmp_path.iterdir()) == []


def test_turns_cancel_only_own_session():
    turns = Turns()
    first = turns.start("a")
    event = turns.get("a", first)
    other = turns.start("b")
    second = turns.start("a")
    assert event.is_set()
    assert not turns.get("a", second).is_set()
    assert not turns.get("b", other).is_set()
    turns.stop("a")
    assert turns.get("a", second).is_set()


def test_phrase_buffering():
    assert split_phrase("Hello") == (None, "Hello")
    assert split_phrase("Hello. Next") == ("Hello.", "Next")
    assert split_phrase("尾句", final=True) == ("尾句", "")
    assert len(split_phrase("x" * 200)[0]) == 180


class Backend:
    def stream(self, messages, cancelled):
        yield "Hello. "
        yield "How are you?"


class Speech:
    def __init__(self, stop=False):
        self.phrases = []
        self.stop = stop

    def generate(self, reference, transcript, phrase, consent, speed, seed, cancelled=None):
        self.phrases.append(phrase)
        if self.stop:
            cancelled.set()
        return "speech.wav", "run.json", {"generation_seconds": .1}


def test_streams_separate_phrases(profile):
    service = Speech()
    cancel = threading.Event()
    events = list(speak_reply(Backend(), service, profile, [], cancel))
    assert service.phrases == ["Hello.", "How are you?"]
    assert sum(e["audio"] is not None for e in events) == 2
    assert events[-1]["first_audio_seconds"] is not None
    assert not cancel.is_set()


def test_cancelled_inflight_audio_never_emitted(profile):
    service = Speech(stop=True)
    events = list(speak_reply(Backend(), service, profile, [], threading.Event()))
    assert all(e["audio"] is None for e in events)
    assert len(service.phrases) == 1


def test_backend_errors_propagate_without_marking_user_cancelled(profile):
    class Broken:
        def stream(self, *args):
            raise RuntimeError("offline")
            yield
    cancel = threading.Event()
    with pytest.raises(RuntimeError, match="offline"):
        list(speak_reply(Broken(), Speech(), profile, [], cancel))
    assert not cancel.is_set()


def test_empty_model_has_actionable_error():
    with pytest.raises(ValueError, match="model name"):
        list(OllamaChat("").stream([], threading.Event()))
