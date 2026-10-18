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
