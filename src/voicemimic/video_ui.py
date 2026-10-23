import os
import gradio as gr
from voicemimic.video import VideoService, validate_clip


def add_video_ui(reference, reference_text, recording, context, main_tabs, capture_steps):
    service = VideoService(os.environ.get('VOICEMIMIC_VIDEO_DIR', 'data/video-imports'))
    imported, excerpt = gr.State(None), gr.State(None)
    gr.Markdown('### 从视频里选一段声音\n粘贴链接 → 识别文字 → 选段试听 → 使用片段。建议选没有配乐、只有一个人说话的完整句子。')
    url = gr.Textbox(label='视频链接', placeholder='https://…')
    load_link = gr.Button('读取链接并识别', variant='primary')
    with gr.Accordion('链接读取失败？上传本地视频或音频', open=False):
        upload = gr.File(label='本地文件 · 最多 300 MB', type='filepath')
        load_file = gr.Button('读取文件并识别')
    gr.Markdown('最多 30 分钟，识别可能需要几分钟。支持情况取决于平台；不支持登录、付费或受保护的视频。')
    status = gr.Textbox(label='当前进度', interactive=False)
    audio = gr.Audio(label='完整音频', interactive=False)
    with gr.Accordion('查看完整识别文字 / 重试识别', open=False):
        full_text = gr.Textbox(label='自动识别文字（可能有误）', lines=6, interactive=False)
        retry = gr.Button('重新识别已读取的音频')
    segment = gr.Dropdown(label='选择一句或多句（自动填入起止时间）', choices=[], multiselect=True)
    gr.Markdown('多句按最早到最晚的时间连续截取，包含中间内容。也可直接调整下面的秒数。')
    with gr.Row():
        start = gr.Number(label='开始（秒）', value=0, precision=2)
        end = gr.Number(label='结束（秒）', value=10, precision=2)
    cut = gr.Button('截取并试听', variant='primary')
    player = gr.Audio(label='选段试听 · 可下载 WAV', interactive=False)
    words = gr.Textbox(label='片段文字（使用前请回听校对）', lines=3)
    gr.Markdown('选段 1–180 秒；用于声音克隆时需 3–12 秒。手动切到句子中间时，请删掉文字中未截入的部分。')
    with gr.Row():
        voice_button = gr.Button('用作声音样本')
        capture_button = gr.Button('送到对话采集')
    with gr.Accordion('来源与保存说明', open=False):
        gr.Markdown('只导入你有权使用的内容，合成前仍需说话者授权。导入不会自动获得审核或授权。音频、转写及片段保存在本机 data/video-imports，需单独清理。')

    def choices(record):
        return [(f"{c['start']:.2f}–{c['end']:.2f}s · {c['text'][:90]}", str(i)) for i, c in enumerate(record['chunks'])]

    def load(link=None, local=None):
        yield None, None, None, '', gr.update(choices=[], value=None), 0, 10, None, '', '正在读取音频…'
        record = None
        try:
            record = service.load(link, local)
            yield record, None, record['audio'], '', gr.update(choices=[], value=None), 0, min(10, record['duration']), None, '', f"已读取：{record['title']}（{record['duration']:.1f} 秒）。正在识别文字…"
            record = service.transcribe(record)
            yield record, None, record['audio'], record['transcript'], gr.update(choices=choices(record), value=None), 0, min(10, record['duration']), None, '', '识别完成，请选一句或手动设置时间。' if record['chunks'] else '未识别到可靠的句子时间，可听完整音频后手动选段。'
        except Exception as exc:
            yield record, None, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), None, '', f'{exc}' + (' 音频已保留，可手动截取或重试识别。' if record else '')

    def from_link(link):
        yield from load(link=link)

    def from_file(local):
        if not local:
            raise gr.Error('请先上传本地视频或音频。')
        yield from load(local=local)

    def retranscribe(record):
        try:
            result = service.transcribe(record)
            return result, result['transcript'], gr.update(choices=choices(result), value=None), '识别完成，可重新选段。'
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def select(record, selection):
        if record is None or not selection:
            return gr.skip(), gr.skip()
        chunks = [record['chunks'][int(i)] for i in selection]
        return min(c['start'] for c in chunks), max(c['end'] for c in chunks)

    def extract(record, begin, finish):
        try:
            result = service.clip(record, begin, finish)
            return result, result['path'], result['text'], '片段已准备好。请试听、校对文字，再选择用途。'
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def use_voice(record, clip, begin, finish, text):
        try:
            path = validate_clip(record, clip, begin, finish, for_voice=True)
            return path, text, gr.update(selected='voice')
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    def use_capture(record, clip, begin, finish):
        try:
            path = validate_clip(record, clip, begin, finish)
            note = f"视频选段：{record['title']}；{begin:.2f}–{finish:.2f} 秒。来源：{record.get('url') or '本地文件'}"
            return path, note[:2000], gr.update(selected='capture'), gr.update(selected='record')
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    outputs = [imported, excerpt, audio, full_text, segment, start, end, player, words, status]
    queue = dict(concurrency_id='video-import', concurrency_limit=1)
    load_link.click(from_link, url, outputs, **queue)
    load_file.click(from_file, upload, outputs, **queue)
    retry.click(retranscribe, imported, [imported, full_text, segment, status], **queue)
    segment.change(select, [imported, segment], [start, end])
    cut.click(extract, [imported, start, end], [excerpt, player, words, status], **queue)
    voice_button.click(use_voice, [imported, excerpt, start, end, words], [reference, reference_text, main_tabs], **queue)
    capture_button.click(use_capture, [imported, excerpt, start, end], [recording, context, main_tabs, capture_steps], **queue)
