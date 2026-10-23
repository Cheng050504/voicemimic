import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from voicemimic.capture import InterviewStore


@pytest.fixture
def interview(tmp_path):
    store = InterviewStore(tmp_path / 'interviews', lambda _: {'text': 'I prefer short answers.', 'chunks': [{'timestamp': [0, 2], 'text': 'I prefer short answers.'}]})
    identifier = store.create('Test participant', True)
    return store, identifier


@pytest.fixture
def sample(interview, tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg required')
    store, identifier = interview
    audio = tmp_path / 'recording.wav'
    sf.write(audio, np.sin(np.arange(48000) * .1) * .2, 16000)
    sid = store.add(identifier, str(audio), 'How do you explain things?', 'test fixture')
    return store, identifier, sid


def test_consent_required(tmp_path):
    with pytest.raises(ValueError):
        InterviewStore(tmp_path).create('Test', False)
    assert list(tmp_path.iterdir()) == []


def test_capture_transcribe_review_and_export(sample):
    store, identifier, sid = sample
    assert store.transcribe(identifier, sid) == 'I prefer short answers.'
    assert json.loads(open(store.export(identifier)).read())['samples'] == []
    store.review(identifier, sid, 'I prefer concise answers.', True)
    report = json.loads(open(store.export(identifier)).read())
    assert report['samples'][0]['text'] == 'I prefer concise answers.'
    assert report['samples'][0]['raw_text'] == 'I prefer short answers.'
    assert report['samples'][0]['timestamps'][0]['timestamp'] == [0, 2]


def proposals(monkeypatch, items):
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'message': {'content': json.dumps({'observations': items})}}
    monkeypatch.setattr('voicemimic.capture.httpx.post', lambda *a, **kw: Response())


def test_observation_grounding_approval_and_invalidation(sample, monkeypatch):
    store, identifier, sid = sample
    store.review(identifier, sid, 'I prefer short answers.', True)
    proposals(monkeypatch, [dict(claim='Prefers short answers', kind='explicit', sample_id=sid, evidence_quote='I prefer short answers.'),
                           dict(claim='Unfounded', kind='inference', sample_id=sid, evidence_quote='not in source')])
    assert store.propose(identifier, 'test') == 1
    assert json.loads(open(store.export(identifier)).read())['observations'] == []
    _, data = store.load(identifier)
    oid = data['observations'][0]['id']
    store.review_observation(identifier, oid, 'Usually prefers short answers', 'explicit', True)
    assert len(json.loads(open(store.export(identifier)).read())['observations']) == 1
    store.review(identifier, sid, 'I prefer detailed answers.', True)
    assert store.load(identifier)[1]['observations'] == []
    assert not (store.load(identifier)[0] / 'approved-profile.json').exists()


def test_delete_removes_source_and_derived_data(sample, monkeypatch):
    store, identifier, sid = sample
    store.review(identifier, sid, 'Hello world.', True)
    proposals(monkeypatch, [dict(claim='Says hello', kind='inference', sample_id=sid, evidence_quote='Hello world.')])
    store.propose(identifier, 'test')
    folder, _, s = store.sample(identifier, sid)
    with pytest.raises(ValueError):
        store.delete(identifier, sid, False)
    assert (folder / s['audio']).exists()
    store.delete(identifier, sid, True)
    assert not (folder / s['audio']).exists()
    assert store.load(identifier)[1]['observations'] == []
    assert store.load(identifier)[1]['samples'] == []


def test_unreviewed_evidence_cannot_be_analyzed(sample):
    store, identifier, sid = sample
    with pytest.raises(ValueError, match='审核'):
        store.propose(identifier, 'test')


def test_invalid_id_rejected(interview):
    store, _ = interview
    with pytest.raises(ValueError):
        store.load('../escape')


def test_voice_excerpt_requires_review_and_tracks_changes(sample):
    store, identifier, sid = sample
    with pytest.raises(ValueError, match='审核'):
        store.voice_excerpt(identifier, sid, 0, 3)
    store.review(identifier, sid, 'I prefer short answers.', True)
    clip = store.voice_excerpt(identifier, sid, 0, 3)
    assert sf.info(clip['path']).duration == 3
    assert clip['text'] == 'I prefer short answers.'
    assert store.voice_reference(identifier, sid, clip, 0, 3, clip['text'])[0] == clip['path']
    with pytest.raises(ValueError, match='选择已改变'):
        store.voice_reference(identifier, sid, clip, 0, 2, clip['text'])
    store.review(identifier, sid, 'Corrected text.', True)
    with pytest.raises(ValueError, match='审核已变化'):
        store.voice_reference(identifier, sid, clip, 0, 3, clip['text'])
    with pytest.raises(ValueError, match='3–12'):
        store.voice_excerpt(identifier, sid, 0, 2)
    from pathlib import Path
    store.delete(identifier, sid, True)
    assert not Path(clip['path']).exists()


def test_new_capture_keeps_sample_rate_and_partial_text_is_not_guessed(interview, tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg required')
    store, identifier = interview
    path = tmp_path/'high_rate.wav'
    sf.write(path, np.sin(np.arange(48000*5)*.1)*.2, 48000)
    sid = store.add(identifier, str(path), 'Question', '')
    folder, _, sample = store.sample(identifier, sid)
    assert sf.info(folder/sample['audio']).samplerate == 48000
    store.review(identifier, sid, 'A complete five-second reply.', True)
    clip=store.voice_excerpt(identifier,sid,1,4)
    assert clip['text'] == ''
    assert sf.info(clip['path']).samplerate == 48000
