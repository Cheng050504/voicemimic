import hashlib
import shutil
import subprocess
import numpy as np
import pytest
import soundfile as sf
from voicemimic.denoise import denoise, checked_preview

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required')


def fixture_audio(path, seconds=4, stereo=False):
    sr = 24000
    t = np.arange(int(seconds * sr)) / sr
    voice = (0.16 * np.sin(2*np.pi*180*t) + 0.06 * np.sin(2*np.pi*360*t)) * ((t > 1) & (t < 3))
    audio = voice + np.random.default_rng(9).normal(0, .008, len(t))
    if stereo:
        audio = np.column_stack([audio, audio * .8])
    sf.write(path, audio, sr, subtype='FLOAT')
    return audio, sr


@pytest.mark.parametrize('preset, noise_ratio', [('轻度', .8), ('保真优先（推荐）', .95)])
def test_reduces_noise_preserves_shape_voice_and_source(tmp_path, preset, noise_ratio):
    source = tmp_path/'input.wav'
    audio, sr = fixture_audio(source, stereo=True)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    result = denoise(source, preset, root=tmp_path/'runs')
    out, out_sr = sf.read(result['cleaned'])
    assert out_sr == sr and out.shape == audio.shape
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    # Settled noise-only tail and harmonic voice energy, not a perceptual identity claim.
    assert np.mean(out[int(3.5*sr):]**2) < np.mean(audio[int(3.5*sr):]**2) * noise_ratio
    segment = slice(int(1.5*sr), int(2.5*sr))
    ratio = np.sqrt(np.mean(out[segment]**2) / np.mean(audio[segment]**2))
    assert .85 < ratio < 1.1
    assert checked_preview(source, result) == result
    assert checked_preview(result['cleaned'], result) == result
    other = tmp_path/'other.wav'
    sf.write(other, np.ones(sr*3)*.02, sr)
    with pytest.raises(ValueError, match='已更换'):
        checked_preview(other, result)


def test_m4a_and_no_silent_trimming(tmp_path):
    source = tmp_path/'input.wav'
    fixture_audio(source)
    m4a = tmp_path/'input.m4a'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source), '-c:a', 'aac', str(m4a)], check=True)
    result = denoise(m4a, root=tmp_path/'runs')
    assert abs(sf.info(result['cleaned']).duration - 4) < .1
    with pytest.raises(ValueError, match='不会自动截短'):
        denoise(source, max_seconds=3, root=tmp_path/'invalid')
    assert list((tmp_path/'invalid').iterdir()) == []


def test_invalid_and_silent_inputs(tmp_path):
    with pytest.raises(ValueError, match='先上传'):
        denoise(None, root=tmp_path/'runs')
    source = tmp_path/'silence.wav'
    sf.write(source, np.zeros(24000*3), 24000)
    with pytest.raises(ValueError, match='太安静'):
        denoise(source, root=tmp_path/'runs')
    assert list((tmp_path/'runs').iterdir()) == []


@pytest.mark.parametrize('sr', [24000, 44100, 48000])
def test_filter_alignment_and_tail_retained(tmp_path, sr):
    source = tmp_path/'pulses.wav'
    audio = np.zeros(sr*3, dtype='float32')
    audio[sr] = .5
    audio[-100] = .5
    sf.write(source,audio,sr,subtype='FLOAT')
    result=denoise(source, '轻度', root=tmp_path/'runs')
    out,_=sf.read(result['cleaned'])
    assert len(out)==len(audio)
    assert abs(np.argmax(np.abs(out[sr-50:sr+50]))-50)<=1
    assert np.max(np.abs(out[-150:])) > .1
