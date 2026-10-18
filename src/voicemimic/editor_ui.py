import gradio as gr


def add_editor(service, job_id, export_outputs, resume, progress):
    def choices(identifier):
        try:
            _, job = service.load(identifier)
            return gr.update(choices=[(f"第 {i+1} 段 · {c['text'][:35]}", i) for i, c in enumerate(job['chunks'])], value=None), '请选择段落进行精修。'
        except ValueError as exc:
            return gr.update(choices=[], value=None), str(exc)

    def details(identifier, index):
        try:
            folder, job, c, target = service.segment(identifier, index)
            versions = c.get('versions', [])
            selected = c.get('selected_version')
            return (c['text'], c.get('spoken_text', c['text']), c.get('speed', job['speed']), c['pause'],
                    c.get('seed', job['seed']), str(target) if target.exists() else None,
                    gr.update(choices=[(f"版本 {n+1} · 语速 {v['speed']} · seed {v['seed']}", v['id']) for n,v in enumerate(versions)], value=selected),
                    '已加载当前段落。改动后请重新合并完整音频。')
        except (ValueError, TypeError) as exc:
            return '', '', 1, 0, 42, None, gr.update(choices=[], value=None), str(exc)

    def preview(identifier, index, version_id):
        try:
            folder, _, c, _ = service.segment(identifier, index)
            v = next(v for v in c.get('versions', []) if v['id'] == version_id)
            return str(folder / v['file'])
        except (ValueError, TypeError, StopIteration):
            return None

    def regenerate(identifier, index, words, rate, pause, seed):
        try:
            service.regenerate(identifier, index, words, rate, pause, seed)
            return (*details(identifier, index)[:-1], '新版本已生成并采用；其他段落未改动。请重新合并。', None, None, None, None)
        except Exception as exc:
            return (*[gr.skip() for _ in range(7)], f'生成失败，原版本保留：{exc}', *[gr.skip() for _ in range(4)])

    def select(identifier, index, version):
        try:
            service.select_version(identifier, index, version)
            return (*details(identifier, index)[:-1], '已采用所选版本，请重新合并。', None, None, None, None)
        except Exception as exc:
            return (*[gr.skip() for _ in range(7)], str(exc), *[gr.skip() for _ in range(4)])

    def pause_only(identifier, index, pause):
        try:
            service.set_pause(identifier, index, pause)
            return '停顿已保存，无需重新生成语音。请重新合并。', None, None, None, None
        except Exception as exc:
            return str(exc), *[gr.skip() for _ in range(4)]

    gr.Markdown('### 配音精修\n使用上方任务编号加载分段。原文保留；「实际读法」用于修正人名、数字或缩写。修改不会自动更新完整音频。')
    load = gr.Button('加载分段列表')
    segment = gr.Dropdown(label='选择要精修的段落', choices=[])
    original = gr.Textbox(label='原文（保留不变）', interactive=False)
    words = gr.Textbox(label='实际读法', lines=3, max_length=600)
    with gr.Row():
        speed = gr.Slider(.7, 1.3, value=1, step=.05, label='本段语速')
        pause = gr.Slider(0, 5, value=.25, step=.05, label='本段之后额外停顿（秒）')
        seed = gr.Number(value=42, precision=0, label='生成 seed（改变可尝试不同效果）')
    with gr.Row():
        regenerate_button = gr.Button('仅重新生成本段', variant='primary')
        pause_button = gr.Button('只保存停顿')
    current = gr.Audio(label='当前采用版本', interactive=False)
    versions = gr.Dropdown(label='历史版本（选择后可试听）', choices=[])
    comparison = gr.Audio(label='历史版本试听', interactive=False)
    select_button = gr.Button('采用此版本')
    status = gr.Textbox(label='精修状态', interactive=False)
    merge = gr.Button('重新合并并导出完整音频')
    details_outputs = [original, words, speed, pause, seed, current, versions, status]
    queue = dict(concurrency_id='narration', concurrency_limit=1)
    job_id.change(choices, [job_id], [segment, status], **queue)
    load.click(choices, [job_id], [segment, status], **queue)
    segment.change(details, [job_id, segment], details_outputs, **queue)
    versions.change(preview, [job_id, segment, versions], comparison, **queue)
    regenerate_button.click(regenerate, [job_id, segment, words, speed, pause, seed], details_outputs + export_outputs, **queue)
    select_button.click(select, [job_id, segment, versions], details_outputs + export_outputs, **queue)
    pause_button.click(pause_only, [job_id, segment, pause], [status] + export_outputs, **queue)
    merge.click(resume, [job_id], [job_id] + [progress] + export_outputs, **queue)
