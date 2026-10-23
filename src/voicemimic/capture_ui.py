import os

import gradio as gr
from voicemimic.denoise_ui import add_denoise_ui

from voicemimic.capture import InterviewStore, QUESTIONS


def add_capture_ui():
    store = InterviewStore(os.environ.get('VOICEMIMIC_INTERVIEW_DIR', 'data/interviews'))
    queue = dict(concurrency_id='capture', concurrency_limit=1)

    def safe(fn):
        def call(*args):
            try:
                return fn(*args)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
        return call

    def create(name, consent):
        identifier = store.create(name, consent)
        return gr.update(choices=store.choices(), value=identifier), '档案已建立，请录制第一条回答。'

    def list_samples(identifier):
        _, data = store.load(identifier)
        return gr.update(choices=[(f"{'✓' if s['approved'] else '待审核'} · {s['question'][:28]}", s['id']) for s in data['samples']], value=None)

    def add(identifier, audio, question, context):
        sample_id = store.add(identifier, audio, question, context)
        update = list_samples(identifier)
        update['value'] = sample_id
        return update, '录音已保存。点击自动转写，或手动填写逐字稿后审核。'

    def show(identifier, sample_id):
        if not sample_id:
            return None, '', False, {}, None
        folder, _, s = store.sample(identifier, sample_id)
        return str(folder / s['audio']), s['text'], s['approved'], dict(sample_id=s['id'], question=s['question'], context=s['context'],
            captured_at=s['captured_at'], duration_seconds=s['duration_seconds'], original_asr_timestamps=s['timestamps']), None

    def transcribe(identifier, sample_id):
        return store.transcribe(identifier, sample_id), False, '转写完成，请听录音核对文字后保存审核。', None

    def review(identifier, sample_id, text, approved):
        store.review(identifier, sample_id, text, approved)
        return '审核已保存；修改来源后，相关观察需重新生成和确认。', None

    def delete(identifier, sample_id, confirmed):
        store.delete(identifier, sample_id, confirmed)
        return list_samples(identifier), None, '', False, {}, None, '样本及相关观察已删除。', False

    def observations(identifier):
        _, data = store.load(identifier)
        return gr.update(choices=[(f"{'✓' if o['approved'] else '待确认'} · {o['claim'][:35]}", o['id']) for o in data['observations']], value=None)

    def propose(identifier, model):
        count = store.propose(identifier, model)
        return observations(identifier), f'新增 {count} 条待确认观察。引用匹配仅验证来源，不证明推测正确。', None

    def show_observation(identifier, oid):
        if not oid:
            return '', 'inference', '', False
        _, data = store.load(identifier)
        o = next((o for o in data['observations'] if o['id'] == oid), None)
        if not o:
            return '', 'inference', '来源已变更，此观察已失效。', False
        return o['claim'], o['kind'], f"样本 {o['sample_id']}\n原文引用：{o['evidence_quote']}", o['approved']

    def source(identifier, oid):
        _, data = store.load(identifier)
        o = next((o for o in data['observations'] if o['id'] == oid), None)
        if not o:
            raise ValueError('请选择仍然有效的观察。')
        return o['sample_id']

    def approve_observation(identifier, oid, claim, kind, approved):
        store.review_observation(identifier, oid, claim, kind, approved)
        return '观察已保存。仅本人确认的观察会进入导出档案。', None

    gr.Markdown("### 从一次自然的回答开始\n建立档案 → 录制回答 → 核对文字 → 确认表达观察。建议先聊 10–15 分钟，不必一次完成。")
    with gr.Tabs() as steps:
        with gr.Tab("1 · 录制回答", id="record"):
            name = gr.Textbox(label='参与者称呼（可以使用昵称）')
            consent = gr.Checkbox(label='参与者同意本机录音、保存、转写及表达分析，并可审核或删除样本。')
            create_button = gr.Button('建立个人档案')
            with gr.Row():
                interview = gr.Dropdown(label='个人档案', choices=store.choices())
                refresh = gr.Button('刷新档案')
            question = gr.Textbox(label='当前问题', value=QUESTIONS[0], lines=2)
            gr.Examples([[q] for q in QUESTIONS[:3]], inputs=[question], label='换个话题（也可修改上方问题）')
            context = gr.Textbox(label='背景备注（选填）')
            recording = gr.Audio(label='回答录音 · 1–180 秒', sources=['upload', 'microphone'], type='filepath')
            add_denoise_ui(recording, 180)
            add_button = gr.Button('保存回答录音', variant='primary')
        with gr.Tab("2 · 核对文字", id="review"):
            gr.Markdown("选择回答，自动转写后回听核对。确认无误，再保存审核状态。")
            sample = gr.Dropdown(label='选择已保存的回答', choices=[])
            reload_samples = gr.Button('刷新回答列表')
            playback = gr.Audio(label='原始回答试听', interactive=False)
            with gr.Accordion("查看来源与时间戳", open=False):
                info = gr.JSON(label='问题、背景、时间及原始转写时间戳')
            asr = gr.Button('自动转写所选回答（首次下载 Whisper 模型）')
            text = gr.Textbox(label='逐字稿（请核对并修正）', lines=6)
            approved = gr.Checkbox(label='本人已核对，并同意此样本用于后续声音与表达学习。')
            review_button = gr.Button('保存逐字稿与审核状态')
            with gr.Accordion("删除回答", open=False):
                delete_confirm = gr.Checkbox(label='确认删除所选回答及其相关观察（本机档案不可恢复）')
                delete_button = gr.Button('删除所选回答')
        with gr.Tab("3 · 确认观察", id="observe"):
            gr.Markdown('### 有证据的表达观察\n只分析已审核文字。本人明确自述与系统推测分开标记；每条都有来源。请本人纠正并确认。不会从音高推断人格。')
            with gr.Accordion("模型设置", open=False):
                model = gr.Textbox(label='本机 Ollama 模型', value=os.environ.get('VOICEMIMIC_CHAT_MODEL', 'qwen2.5:1.5b'))
            propose_button = gr.Button('从已审核回答生成待确认观察')
            obs = gr.Dropdown(label='表达观察', choices=[])
            refresh_obs = gr.Button('刷新观察列表')
            claim = gr.Textbox(label='观察内容（本人可修正）', lines=3)
            kind = gr.Radio([('本人明确自述', 'explicit'), ('系统推测', 'inference')], value='inference', label='证据类型（请核对分类）')
            evidence = gr.Textbox(label='来源证据', lines=3, interactive=False)
            source_button = gr.Button('打开该观察的来源回答')
            confirmed = gr.Checkbox(label='本人确认此条观察及其分类。')
            approve = gr.Button('保存观察审核')
            export_button = gr.Button('导出已审核档案 JSON')
            export = gr.File(label='已审核档案（不包含未审核内容）', interactive=False)
    status = gr.Textbox(label='当前进度', interactive=False)
    with gr.Accordion("数据与隐私说明", open=False):
        gr.Markdown('数据保存在本机 data/interviews。转写错误需人工核对；改字后原始时间戳不会自动重新对齐。删除会移除本机档案文件和相关观察，但已经下载的副本及浏览器/Gradio 临时缓存需另行清理。')

    create_button.click(safe(create), [name, consent], [interview, status], **queue)
    refresh.click(lambda: gr.update(choices=store.choices()), outputs=interview, **queue)
    interview.change(safe(list_samples), [interview], sample, **queue)
    interview.change(safe(observations), [interview], obs, **queue)
    add_button.click(safe(add), [interview, recording, question, context], [sample, status], **queue).success(lambda: gr.update(selected='review'), outputs=steps)
    reload_samples.click(safe(list_samples), [interview], sample, **queue)
    sample.change(safe(show), [interview, sample], [playback, text, approved, info, export], **queue)
    asr.click(safe(transcribe), [interview, sample], [text, approved, status, export], **queue)
    review_button.click(safe(review), [interview, sample, text, approved], [status, export], **queue)
    delete_button.click(safe(delete), [interview, sample, delete_confirm], [sample, playback, text, approved, info, export, status, delete_confirm], **queue)
    propose_button.click(safe(propose), [interview, model], [obs, status, export], **queue)
    refresh_obs.click(safe(observations), [interview], obs, **queue)
    obs.change(safe(show_observation), [interview, obs], [claim, kind, evidence, confirmed], **queue)
    source_button.click(safe(source), [interview, obs], sample, **queue).success(lambda: gr.update(selected='review'), outputs=steps)
    approve.click(safe(approve_observation), [interview, obs, claim, kind, confirmed], [status, export], **queue)
    export_button.click(safe(store.export), [interview], export, **queue)
