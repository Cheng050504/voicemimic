import functools
import http.server
import json
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voicemimic.video import VideoService, validate_clip, validate_url

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required')


def video_fixture(tmp_path):
    wav = tmp_path/'speech.wav'
    t = np.arange(6*24000)/24000
    sf.write(wav, .15*np.sin(2*np.pi*220*t), 24000)
    video = tmp_path/'sample.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=64x64:r=5',
                    '-i', str(wav), '-t', '6', '-c:v', 'mpeg4', '-c:a', 'aac', str(video)], check=True)
    return video


def fake_asr(path):
    assert sf.info(path).samplerate == 16000
    return dict(text='第一句。第二句。', chunks=[{'text':'第一句。','timestamp':(0,3)}, {'text':'第二句。','timestamp':(3,None)}])


def test_local_video_transcript_excerpt_and_adoption(tmp_path):
    source = video_fixture(tmp_path)
    service = VideoService(tmp_path/'imports', fake_asr)
    record = service.transcribe(service.load(upload=source))
    assert source.exists()
    assert record['chunks'][1]['end'] == record['duration']
    clip = service.clip(record, 1, 5)
    assert sf.info(clip['path']).frames == 4*48000
    assert clip['text'] == '第一句。 第二句。'
    assert validate_clip(record, clip, 1, 5, True) == clip['path']
    with pytest.raises(ValueError, match='已改变'):
        validate_clip(record, clip, 2, 5)
    with pytest.raises(ValueError, match='已改变'):
        validate_clip(dict(record,id='other'), clip, 1, 5)
    short = service.clip(record, 1, 2)
    with pytest.raises(ValueError, match='3–12'):
        validate_clip(record, short, 1, 2, True)
    for start,end in [(float('nan'),5),(-1,4),(4,2),(0,99)]:
        with pytest.raises(ValueError):
            service.clip(record,start,end)


def test_link_download_real_ytdlp(tmp_path):
    pytest.importorskip('yt_dlp')
    source = video_fixture(tmp_path)
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*args):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1',0), functools.partial(Handler,directory=str(tmp_path)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        url=f'http://127.0.0.1:{server.server_port}/{source.name}'
        record=VideoService(tmp_path/'imports',fake_asr).load(url=url)
        assert record['url']==url
        assert 6 <= record['duration'] < 6.1
        folder=Path(record['audio']).parent
        assert sorted(p.name for p in folder.iterdir()) == ['audio.wav','source.json']
    finally:
        server.shutdown();server.server_close();thread.join()


def test_failed_import_cleanup_and_url_validation(tmp_path):
    bad=tmp_path/'bad.mp4';bad.write_text('not a video')
    service=VideoService(tmp_path/'imports')
    with pytest.raises(ValueError):
        service.load(upload=bad)
    assert list((tmp_path/'imports').iterdir())==[]
    for url in ['', 'file:///etc/passwd', 'https://user:pass@example.com/a', 'javascript:alert(1)']:
        with pytest.raises(ValueError):
            validate_url(url)


def test_overlong_clip_rejected(tmp_path):
    path=tmp_path/'audio.wav'
    sf.write(path,np.zeros(200*16000),16000)
    record=dict(id='test',audio=str(path),duration=200,chunks=[])
    with pytest.raises(ValueError,match='1–180'):
        VideoService(tmp_path/'imports').clip(record,0,181)
