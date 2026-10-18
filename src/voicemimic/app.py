import logging
import os

# Disable Gradio analytics before importing it; inference stays on this machine.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

from voicemimic.core import VoiceService, inspect_audio
from voicemimic.chat_ui import add_chat_ui
from voicemimic.narration_ui import add_narration_ui

EXAMPLES = [
    "I wasn't expecting that, but I'd like to hear more.",
    "Could you explain what you mean by that?",
    "Let's take a step back and try a different approach.",
    "The meeting starts at nine tomorrow morning.",
    "That was a surprisingly good cup of coffee.",
    "I agree with part of your point, but I see it differently.",
    "Take your time. We don't have to decide right now.",
    "First, open the window. Then turn the handle to the left.",
    "Really? I thought you were joking!",
    "I've been thinking about our conversation since yesterday.",
]


def build_app(service=None):
    service = service or VoiceService(os.environ.get("VOICEMIMIC_OUTPUT_DIR", "data/runs"))

    def inspect(reference):
        if not reference:
            return "Add a reference recording to check its quality."
        try:
            _, _, q = inspect_audio(reference)
            return f"{q['duration_seconds']} seconds · {q['sample_rate']} Hz. " + " ".join(q["warnings"])
        except ValueError as exc:
            return str(exc)

    def generate(*args):
        # Clear old outputs immediately so failures cannot be mistaken for a new result.
        yield None, None, "Loading voice engine / generating audio. First use downloads model weights."
        try:
            audio, manifest, metadata = service.generate(*args)
            yield audio, manifest, f"Complete in {metadata['generation_seconds']}s. Compare the recordings below."
        except (ValueError, RuntimeError) as exc:
            yield None, None, str(exc)
        except Exception:
            logging.exception("Voice generation failed")
            yield None, None, "Generation failed. Check terminal logs for model download or device errors."

    with gr.Blocks(title="VoiceMimic · Timbre Lab", delete_cache=(3600, 86400)) as app:
        gr.Markdown("# VoiceMimic\n### Timbre Lab · Milestone 01\nMake new speech from a short voice reference, then listen side by side.")
        gr.Markdown("Local research prototype · F5-TTS weights are noncommercial · English / Mandarin baseline")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Reference voice")
                reference = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference · 3–12 seconds")
                gr.Markdown("One speaker, no background music. Finish the sentence and leave a short pause. Upload WAV, FLAC, or M4A.")
                quality = gr.Textbox(label="Recording check", interactive=False)
                transcript = gr.Textbox(label="Exact words in the reference", lines=3, placeholder="Type every spoken word, including fillers.")
                consent = gr.Checkbox(label="This is my voice, or I have the speaker's permission to synthesize it.")
            with gr.Column():
                gr.Markdown("### 2. New speech")
                text = gr.Textbox(label="What should the voice say?", lines=4, max_length=600)
                gr.Examples([[x] for x in EXAMPLES], inputs=[text], label="Held-out sentence ideas")
                with gr.Accordion("Generation settings", open=False):
                    speed = gr.Slider(0.7, 1.3, value=1, step=0.05, label="Speaking speed")
                    seed = gr.Number(value=42, precision=0, label="Seed · reuse for comparisons")
                button = gr.Button("Generate speech", variant="primary")
        gr.Markdown("### 3. Listen and compare\nPlay the reference above, then the generated audio. Rate resemblance, pronunciation, and naturalness separately.")
        output = gr.Audio(label="AI-generated speech", type="filepath", interactive=False)
        status = gr.Textbox(label="Status", interactive=False)
        manifest = gr.File(label="Download experiment record", interactive=False)
        gr.Markdown("Runs stay in `data/runs` until you delete them. They contain audio and transcripts. Model weights download on first use; reference audio is processed locally. This app does not verify permission or watermark audio.")
        add_chat_ui(service, reference, transcript, consent, speed, seed)
        add_narration_ui(service)
        reference.change(inspect, inputs=[reference], outputs=[quality])
        button.click(generate, inputs=[reference, transcript, text, consent, speed, seed],
                     outputs=[output, manifest, status], concurrency_limit=1)
    return app


def main():
    build_app().queue(max_size=4).launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
