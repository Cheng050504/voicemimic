import shutil

import numpy as np
import pytest
import soundfile as sf

from voicemimic.narration import NarrationService, split_script


@pytest.fixture
def profile(tmp_path):
    ref = tmp_path / 'reference.wav'
    sf.write(ref, np.sin(np.arange(24000) * .1) * .2, 24000)
    return dict(reference=str(ref), name='Test', id='a' * 32, transcript='Hello.',
                speed=1, seed=42, permission_confirmed=True)


class Speech:
    def __init__(self, profile, fail_on=None):
        self.path = profile['reference']
        self.calls = []
        self.fail_on = fail_on

    def generate(self, reference, transcript, text, *args):
        self.calls.append(text)
        if len(self.calls) == self.fail_on:
            raise RuntimeError('temporary failure')
        return self.path, None, {}


def test_split_preserves_words_and_chinese():
    text = ('这是一个比较长的句子，没有任何空格。' * 40) + '\n\n' + ('Hello world! ' * 60)
    parts = split_script(text)
    assert all(0 < len(p['text']) <= 240 for p in parts)
    assert ''.join(''.join(p['text'].split()) for p in parts) == ''.join(text.split())
    assert parts[-1]['pause'] == 0
    assert any(p['pause'] == .65 for p in parts[:-1])


@pytest.mark.parametrize('text', ['', ' ', 'x' * 10001])
def test_bad_scripts(text):
    with pytest.raises(ValueError):
        split_script(text)


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg export integration')
def test_full_audio_and_resume_after_failed_part(tmp_path, profile):
    speech = Speech(profile, fail_on=2)
    service = NarrationService(speech, tmp_path / 'jobs')
    job_id = service.create('First paragraph.\n\nSecond paragraph.', profile)
    with pytest.raises(RuntimeError, match='temporary'):
        list(service.run(job_id))
    assert service.load(job_id)[1]['status'] == 'failed'
    speech.fail_on = None
    events = list(service.run(job_id))
    assert speech.calls == ['First paragraph.', 'Second paragraph.', 'Second paragraph.']
    final = events[-1]
    assert sf.info(final['wav']).duration == pytest.approx(2.65)
    assert sf.info(final['mp3']).duration > 2.5
    assert service.load(job_id)[1]['status'] == 'complete'
    # Reopening a completed task does not resynthesize any segments.
    list(service.run(job_id))
    assert len(speech.calls) == 3


def test_job_freezes_profile_and_rejects_paths(tmp_path, profile):
    service = NarrationService(Speech(profile), tmp_path / 'jobs')
    job_id = service.create('Script.', profile)
    profile['speed'] = 1.3
    assert service.load(job_id)[1]['speed'] == 1
    with pytest.raises(ValueError):
        service.load('../escape')


def test_wav_survives_mp3_export_failure(tmp_path, profile, monkeypatch):
    service = NarrationService(Speech(profile), tmp_path / 'jobs')
    job_id = service.create('Script.', profile)
    monkeypatch.setattr(shutil, 'which', lambda _: None)
    events = []
    with pytest.raises(RuntimeError, match='FFmpeg'):
        for event in service.run(job_id):
            events.append(event)
    assert sf.info(events[-1]['wav']).duration == 1


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg export integration')
def test_edit_one_segment_versions_and_remerge(tmp_path, profile):
    speech = Speech(profile)
    service = NarrationService(speech, tmp_path / 'jobs')
    job_id = service.create('Original first.\n\nOriginal second.', profile)
    list(service.run(job_id))
    folder, _ = service.load(job_id)
    second = (folder / 'part-0001.wav').read_bytes()
    version = service.regenerate(job_id, 0, 'Corrected pronunciation.', 1.1, 1.5, 9)
    _, job = service.load(job_id)
    chunk = job['chunks'][0]
    assert chunk['text'] == 'Original first.'
    assert chunk['spoken_text'] == 'Corrected pronunciation.'
    assert len(chunk['versions']) == 2
    assert chunk['selected_version'] == version
    assert (folder / 'part-0001.wav').read_bytes() == second
    assert len(speech.calls) == 3
    final = list(service.run(job_id))[-1]
    assert sf.info(final['wav']).duration == pytest.approx(3.5)
    assert len(speech.calls) == 3
    service.select_version(job_id, 0, 'v0')
    _, job = service.load(job_id)
    assert job['chunks'][0]['spoken_text'] == 'Original first.'
    assert job['chunks'][0]['speed'] == 1
    service.set_pause(job_id, 0, 2)
    assert service.load(job_id)[1]['chunks'][0]['pause'] == 2
    assert len(speech.calls) == 3


def test_failed_regeneration_preserves_original(tmp_path, profile):
    speech = Speech(profile)
    service = NarrationService(speech, tmp_path / 'jobs')
    job_id = service.create('Original.', profile)
    folder, _ = service.load(job_id)
    shutil.copyfile(profile['reference'], folder / 'part-0000.wav')
    before = (folder / 'job.json').read_bytes()
    original = (folder / 'part-0000.wav').read_bytes()
    speech.fail_on = 1
    with pytest.raises(RuntimeError):
        service.regenerate(job_id, 0, 'New.', 1, .5, 42)
    assert (folder / 'job.json').read_bytes() == before
    assert (folder / 'part-0000.wav').read_bytes() == original


@pytest.mark.parametrize('pause', [-1, 6, float('nan')])
def test_invalid_edit_pause(tmp_path, profile, pause):
    service = NarrationService(Speech(profile), tmp_path / 'jobs')
    job_id = service.create('Original.', profile)
    with pytest.raises(ValueError):
        service.set_pause(job_id, 0, pause)
