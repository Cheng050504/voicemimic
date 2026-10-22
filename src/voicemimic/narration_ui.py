import os
import logging

import gradio as gr

from voicemimic.narration import NarrationService
from voicemimic.profiles import ProfileStore
from voicemimic.editor_ui import add_editor


def add_narration_ui(voice_service):
    store = ProfileStore(os.environ.get("VOICEMIMIC_PROFILE_DIR", "data/voices"))
    service = NarrationService(voice_service, os.environ.get("VOICEMIMIC_NARRATION_DIR", "data/narrations"))

    def render(job_id):
        for event in service.run(job_id):
            yield (job_id, event["status"], event.get("wav", gr.skip()),
                   event.get("wav", gr.skip()), event.get("mp3", gr.skip()), event.get("manifest", gr.skip()))

    def start(text, voice_id):
        job_id = ""
        yield "", "正在准备…", None, None, None, None
        try:
            job_id = service.create(text, store.load(voice_id))
            yield job_id, "任务已保存，开始配音…", None, None, None, None
            yield from render(job_id)
        except Exception as exc:
            logging.exception("Narration failed")
            yield job_id, f"生成未完成：{exc} 已完成的分段会保留，可继续 / 重试。", gr.skip(), gr.skip(), gr.skip(), gr.skip()

    def resume(job_id):
        yield job_id, "正在恢复原任务（使用原文案及声音设置）…", None, None, None, None
        try:
            yield from render(job_id)
        except Exception as exc:
            logging.exception("Narration resume failed")
            yield job_id, f"未完成：{exc} 可再次重试。", gr.skip(), gr.skip(), gr.skip(), gr.skip()

    gr.Markdown("### 文案变成完整配音\n① 选择声音　② 粘贴文案　③ 生成并下载。还没有声音？先到「我的声音」保存一个。")
    with gr.Row():
        voice = gr.Dropdown(choices=store.choices(), label="配音声音")
        refresh = gr.Button("刷新声音列表")
    text = gr.Textbox(label="文案", lines=10, max_length=10000,
                      placeholder="粘贴文案。空行表示段落间的停顿；最多 10,000 个字符。")
    gr.Markdown("长文案会按句子和段落自动分段；较长的句子也会切分。长文案可能需要几分钟。")
    generate = gr.Button("生成完整配音", variant="primary")
    with gr.Accordion("继续上次任务", open=False):
        job_id = gr.Textbox(label="任务编号（可保存此编号，重开页面后继续）")
        retry = gr.Button("继续 / 重试原任务")
        gr.Markdown("仅补做未完成的分段，沿用原文案和声音。修改文案后，请生成新任务。")
    status = gr.Textbox(label="配音进度", interactive=False)
    player = gr.Audio(label="完整配音试听 · AI 生成", interactive=False)
    with gr.Row():
        wav = gr.File(label="下载 WAV", interactive=False)
        mp3 = gr.File(label="下载 MP3", interactive=False)
    with gr.Accordion("任务记录", open=False):
        manifest = gr.File(label="下载任务记录", interactive=False)
        gr.Markdown("音频与文案保存在本机 data/narrations。")
    refresh.click(lambda: gr.update(choices=store.choices()), outputs=voice)
    outputs = [job_id, status, player, wav, mp3, manifest]
    generate.click(start, [text, voice], outputs, concurrency_id="narration", concurrency_limit=1)
    retry.click(resume, [job_id], outputs, concurrency_id="narration", concurrency_limit=1)

    with gr.Accordion("需要改某一句？打开逐段精修", open=False):
        add_editor(service, job_id, [player, wav, mp3, manifest], resume, status)
