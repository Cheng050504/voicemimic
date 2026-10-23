import logging
import os

# Disable Gradio analytics before importing it; inference stays on this machine.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr
from voicemimic.denoise_ui import add_denoise_ui
from voicemimic.video_ui import add_video_ui

from voicemimic.core import VoiceService, inspect_audio
from voicemimic.chat_ui import add_chat_ui
from voicemimic.narration_ui import add_narration_ui
from voicemimic.capture_ui import add_capture_ui

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
            return "添加录音后，会自动检查质量。"
        try:
            _, _, q = inspect_audio(reference)
            return f"{q['duration_seconds']} seconds · {q['sample_rate']} Hz. " + " ".join(q["warnings"])
        except ValueError as exc:
            return str(exc)

    def generate(*args):
        # Clear old outputs immediately so failures cannot be mistaken for a new result.
        yield None, None, "正在加载模型并生成试音；首次使用需要下载模型。"
        try:
            audio, manifest, metadata = service.generate(*args)
            yield audio, manifest, f"已完成，用时 {metadata['generation_seconds']} 秒。请试听并对比原录音。"
        except (ValueError, RuntimeError) as exc:
            yield None, None, str(exc)
        except Exception:
            logging.exception("Voice generation failed")
            yield None, None, "生成失败，请查看终端中的模型下载或设备错误。"

    with gr.Blocks(title="VoiceMimic · Timbre Lab", delete_cache=(3600, 86400)) as app:
        gr.Markdown("# VoiceMimic\n从真实对话，建立你的声音与表达档案。")
        gr.Markdown("**第一次使用？** 从「对话采集」开始。已有声音想生成音频，请选「配音与试听」。")
        with gr.Tabs() as main_tabs:
            with gr.Tab("对话采集", id="capture"):
                recording, capture_context, capture_steps = add_capture_ui()
            with gr.Tab("我的声音", id="voice"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 1. 添加声音样本")
                        reference = gr.Audio(sources=["upload", "microphone"], type="filepath", label="声音样本 · 3–12 秒")
                        gr.Markdown("上传清晰的单人录音（WAV / FLAC / M4A），保留完整句子。")
                        add_denoise_ui(reference, 12)
                        quality = gr.Textbox(label="录音检查", interactive=False)
                        transcript = gr.Textbox(label="录音里说了什么？", lines=3, placeholder="填写与录音一致的文字，包含语气词。")
                        consent = gr.Checkbox(label="这是我的声音，或我已取得说话者的合成授权。")
                    with gr.Column():
                        gr.Markdown("### 2. 试一句新话")
                        text = gr.Textbox(label="试音文案", lines=4, max_length=600)
                        gr.Examples([[x] for x in EXAMPLES[:3]], inputs=[text], label="试音例句")
                        with gr.Accordion("更多设置", open=False):
                            speed = gr.Slider(0.7, 1.3, value=1, step=0.05, label="语速")
                            seed = gr.Number(value=42, precision=0, label="随机种子（相同值便于对比）")
                        button = gr.Button("生成试音", variant="primary")
                gr.Markdown("### 3. 听听效果\n对比原录音与生成结果。满意后，在下方保存声音。")
                output = gr.Audio(label="生成的试音", type="filepath", interactive=False)
                status = gr.Textbox(label="生成进度", interactive=False)
                with gr.Accordion("生成记录", open=False):
                    manifest = gr.File(label="下载生成记录", interactive=False)
                gr.Markdown("音频在本机处理；首次生成需加载模型，请稍等。")
                add_chat_ui(service, reference, transcript, consent, speed, seed)
            with gr.Tab("配音与试听", id="narration"):
                add_narration_ui(service)
            with gr.Tab("视频选段", id="video"):
                add_video_ui(reference, transcript, recording, capture_context, main_tabs, capture_steps)
        reference.change(inspect, inputs=[reference], outputs=[quality])
        button.click(generate, inputs=[reference, transcript, text, consent, speed, seed],
                     outputs=[output, manifest, status], concurrency_limit=1)
    return app


def main():
    build_app().queue(max_size=4).launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
