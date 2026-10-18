"""Streaming local chat, phrase buffering, and cooperative speech cancellation."""
import json
import queue
import re
import threading
import time
import uuid

import httpx


class OllamaChat:
    def __init__(self, model, base_url="http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def stream(self, messages, cancelled):
        if not self.model.strip():
            raise ValueError("Enter an installed Ollama model name in chat settings.")
        try:
            with httpx.stream("POST", self.base_url + "/api/chat", json={
                "model": self.model, "messages": messages, "stream": True,
                "options": {"num_predict": 180},
            }, timeout=httpx.Timeout(60, connect=5)) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if cancelled.is_set():
                        return
                    if line:
                        event = json.loads(line)
                        if event.get("error"):
                            raise RuntimeError(event["error"])
                        text = event.get("message", {}).get("content", "")
                        if text:
                            yield text
                        if event.get("done"):
                            return
        except httpx.HTTPStatusError as exc:
            raise RuntimeError("Ollama rejected the request. Check the installed model name.") from exc
        except httpx.RequestError as exc:
            raise RuntimeError("Cannot reach Ollama. Start Ollama locally and install a chat model; see README.") from exc


class Turns:
    """Each browser session owns one active turn. New turns invalidate old audio."""
    def __init__(self):
        self.lock = threading.Lock()
        self.active = {}

    def start(self, session):
        with self.lock:
            old = self.active.pop(session, None)
            if old:
                old[1].set()
            token, cancelled = uuid.uuid4().hex, threading.Event()
            self.active[session] = (token, cancelled)
            return token

    def get(self, session, token):
        with self.lock:
            current = self.active.get(session)
            if current and current[0] == token:
                return current[1]
        stopped = threading.Event()
        stopped.set()
        return stopped

    def stop(self, session):
        with self.lock:
            current = self.active.pop(session, None)
            if current:
                current[1].set()


def split_phrase(buffer, final=False):
    """Prefer sentence boundaries; cap long punctuation-free phrases at 180 chars."""
    match = re.search(r'[.!?。！？](?:\s|$)', buffer)
    if match:
        end = match.end()
    elif len(buffer) >= 180:
        end = buffer.rfind(" ", 0, 180)
        if end < 1:
            end = 180
    elif final:
        end = len(buffer)
    else:
        return None, buffer
    return buffer[:end].strip(), buffer[end:].lstrip()


def speak_reply(backend, service, profile, messages, cancelled):
    """Text generation continues in a bounded queue while each phrase is synthesized.

    Cancellation suppresses in-flight synthesis results; it cannot preempt a GPU kernel.
    Events contain cumulative text, optional audio path, and wall-clock metrics.
    """
    events = queue.Queue(maxsize=64)
    started = time.monotonic()
    producer_stopped = threading.Event()

    class ProducerCancellation:
        def is_set(self):
            return cancelled.is_set() or producer_stopped.is_set()

    producer_cancelled = ProducerCancellation()

    def put(item):
        while not producer_cancelled.is_set():
            try:
                events.put(item, timeout=.1)
                return
            except queue.Full:
                continue

    def produce():
        try:
            for text in backend.stream(messages, producer_cancelled):
                if cancelled.is_set():
                    break
                put(("text", text))
        except Exception as exc:
            put(("error", exc))
        finally:
            put(("end", None))

    threading.Thread(target=produce, daemon=True).start()
    buffer, full_text = "", ""
    first_audio = None
    audio_seconds = 0.0
    finished = False
    try:
        while not cancelled.is_set():
            phrase, rest = split_phrase(buffer, final=finished)
            if phrase:
                buffer = rest
                if cancelled.is_set():
                    return
                audio, _, metadata = service.generate(
                    profile["reference"], profile["transcript"], phrase, True,
                    profile["speed"], profile["seed"], cancelled=cancelled,
                )
                if cancelled.is_set():
                    return
                if first_audio is None:
                    first_audio = round(time.monotonic() - started, 2)
                audio_seconds += metadata.get("generation_seconds", 0)
                yield {"text": full_text, "audio": audio, "first_audio_seconds": first_audio,
                       "generation_seconds": round(audio_seconds, 2)}
                continue
            if finished:
                if not full_text.strip():
                    raise RuntimeError("The chat model returned no spoken text.")
                return
            try:
                kind, value = events.get(timeout=.1)
            except queue.Empty:
                continue
            if kind == "error":
                raise value
            if kind == "end":
                finished = True
            else:
                full_text += value
                buffer += value
                yield {"text": full_text, "audio": None, "first_audio_seconds": first_audio,
                       "generation_seconds": round(audio_seconds, 2)}
    finally:
        producer_stopped.set()
