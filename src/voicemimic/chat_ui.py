"""已保存的声音s and text-in / streamed-audio-out controls."""
import logging
import os

import gradio as gr

from voicemimic.chat import OllamaChat, Turns, speak_reply
from voicemimic.profiles import ProfileStore

# Pause buffered playback immediately, before the server handles cancellation.
STOP_AUDIO_JS = """() => {
  document.querySelectorAll('#chat-speech audio').forEach(a => {
    a.pause(); a.removeAttribute('src'); a.load();
  });
}"""


def add_chat_ui(service, reference, transcript, consent, speed, seed):
    store = ProfileStore(os.environ.get("VOICEMIMIC_PROFILE_DIR", "data/voices"))
    turns = Turns()

    def save(name, path, words, allowed, rate, random_seed):
        try:
            profile = store.save(name, path, words, allowed, rate, random_seed)
            return gr.update(choices=store.choices(), value=profile["id"]), "Voice saved locally."
        except ValueError as exc:
            return gr.skip(), str(exc)

    def load(voice_id):
        try:
            p = store.load(voice_id)
            return p["reference"], p["transcript"], True, p["speed"], p["seed"], f"Loaded {p['name']}."
        except ValueError as exc:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), str(exc)

    def begin(request: gr.Request):
        return turns.start(request.session_hash), None, "Starting reply…"

    def stop(request: gr.Request):
        turns.stop(request.session_hash)
        return None, "Stopped. In-flight computation may finish, but its audio is discarded."

    def reply(message, voice_id, model, history, token, request: gr.Request):
        cancelled = turns.get(request.session_hash, token)
        if cancelled.is_set():
            return
        history = list(history or [])
        try:
            if not message or not message.strip() or len(message) > 2000:
                raise ValueError("Enter a message of 1–2,000 characters.")
            profile = store.load(voice_id)
            conversation = history + [{"role": "user", "content": message.strip()}]
            messages = [{"role": "system", "content":
                         "You are a helpful assistant. Reply in one or two short sentences of plain spoken text. "
                         "Do not claim to be the person whose voice is being synthesized."}] + conversation[-20:]
            response = ""
            for event in speak_reply(OllamaChat(model), service, profile, messages, cancelled):
                if cancelled.is_set():
                    return
                response = event["text"]
                display = conversation + [{"role": "assistant", "content": response}]
                timing = event["first_audio_seconds"]
                status = (f"First audio ready: {timing}s · speech generation: {event['generation_seconds']}s"
                          if timing is not None else "Receiving reply / preparing first speech phrase…")
                yield display, gr.skip(), event["audio"] if event["audio"] else gr.skip(), status
            if response and not cancelled.is_set():
                completed = conversation + [{"role": "assistant", "content": response}]
                yield completed, completed, gr.skip(), status + " · Complete (server readiness, not playback latency)."
        except Exception as exc:
            if cancelled.is_set():
                return
            logging.exception("Chat reply failed")
            yield gr.skip(), gr.skip(), None, str(exc)
        finally:
            cancelled.set()

    gr.Markdown("### 保存与复用")
    gr.Markdown("给上方声音取个名字，保存后就能在配音中选择。")
    with gr.Row():
        name = gr.Textbox(label="声音名称", placeholder="例如：我的日常声音")
        save_button = gr.Button("保存这个声音")
    with gr.Row():
        voices = gr.Dropdown(choices=store.choices(), label="已保存的声音")
        refresh = gr.Button("刷新列表")
        load_button = gr.Button("载入声音")
    profile_status = gr.Textbox(label="保存状态", interactive=False)
    with gr.Accordion("与这个声音聊天（实验功能，生成较慢）", open=False):
        with gr.Accordion("聊天模型设置", open=False):
            model = gr.Textbox(label="本机模型名称", value=os.environ.get("VOICEMIMIC_CHAT_MODEL", ""),
                               placeholder="Enter a model name from ollama list")
            gr.Markdown("Connects to Ollama at localhost:11434. Start Ollama and install a model first. Chat history stays in this browser session; speech experiment files remain on disk.")
        history = gr.State([])
        token = gr.State("")
        conversation = gr.Chatbot(label="对话", height=300)
        message = gr.Textbox(label="你想说什么？", max_length=2000)
        with gr.Row():
            send = gr.Button("发送并朗读", variant="primary")
            stop_button = gr.Button("停止")
        speech = gr.Audio(label="语音回复", streaming=True, autoplay=True, interactive=False,
                          format="wav", elem_id="chat-speech")
        status = gr.Textbox(label="对话进度", interactive=False)
        gr.Markdown("If autoplay is blocked, press Play. Stop clears buffered speech immediately; an active model calculation may continue until the current phrase finishes.")

    save_button.click(save, [name, reference, transcript, consent, speed, seed], [voices, profile_status])
    refresh.click(lambda: gr.update(choices=store.choices()), outputs=[voices])
    load_button.click(load, [voices], [reference, transcript, consent, speed, seed, profile_status])
    started = gr.on([send.click, message.submit], begin, outputs=[token, speech, status],
                    queue=False, js=STOP_AUDIO_JS)
    answering = started.then(reply, [message, voices, model, history, token],
                              [conversation, history, speech, status], concurrency_limit=None)
    stop_button.click(stop, outputs=[speech, status], queue=False, cancels=[answering], js=STOP_AUDIO_JS)
