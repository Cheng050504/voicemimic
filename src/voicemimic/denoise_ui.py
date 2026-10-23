import os
import gradio as gr
from voicemimic.denoise import denoise, checked_preview, fingerprint


def add_denoise_ui(recording, max_seconds):
    state = gr.State(None)

    def preview(current, preset, previous):
        yield gr.skip(), None, None, '正在处理，原录音不会被覆盖…'
        try:
            # Adjust strength from the untouched original, never stack filters.
            if previous and fingerprint(current) == previous['cleaned_hash']:
                current = previous['original']
            result = denoise(current, preset, max_seconds, os.environ.get('VOICEMIMIC_DENOISE_DIR', 'data/denoise'))
            # Keep adoption linked to the currently selected recording too.
            if previous and current == previous['original']:
                result['source_hash'] = previous['cleaned_hash']
            yield result, result['original'], result['cleaned'], '请对比试听。满意后点击「使用降噪版」，否则保留原音。'
        except Exception as exc:
            yield None, None, None, str(exc)

    def adopt(current, result, use_original):
        try:
            checked_preview(current, result)
            return result['original' if use_original else 'cleaned'], ('已恢复原音。' if use_original else '已使用降噪版，可继续保存或生成试音。')
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Accordion('录音有杂音？先试听降噪（可选）', open=False):
        gr.Markdown('先试保真优先：混合原音与轻度降噪，宁可留些底噪，也少损失气声。适合持续底噪；人声重叠、音乐和爆音未必能修好。')
        strength = gr.Radio(['保真优先（推荐）', '轻度', '标准'], value='保真优先（推荐）', label='降噪强度')
        process = gr.Button('预览降噪')
        with gr.Row():
            original = gr.Audio(label='处理前', interactive=False)
            cleaned = gr.Audio(label='处理后 · 可下载', interactive=False)
        with gr.Row():
            apply = gr.Button('使用降噪版')
            restore = gr.Button('恢复原音')
        status = gr.Textbox(label='降噪状态', interactive=False)
        gr.Markdown('仅点击「使用降噪版」才影响后续操作。原文件与处理副本保存在本机 data/denoise，不随档案删除自动清理。')
    process.click(preview, [recording, strength, state], [state, original, cleaned, status], concurrency_limit=1)
    apply.click(lambda current, result: adopt(current, result, False), [recording, state], [recording, status])
    restore.click(lambda current, result: adopt(current, result, True), [recording, state], [recording, status])
