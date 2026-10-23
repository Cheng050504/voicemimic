"""Video-link audio import and sample-accurate local excerpts."""
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import soundfile as sf
from voicemimic.capture import Transcriber

MAX_SECONDS = 1800
MAX_BYTES = 300 * 1024 * 1024


def command(args, timeout=240, accepted_codes=(0,)):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise ValueError('读取超时，请使用更短的视频或上传本地文件。') from exc
    if proc.returncode not in accepted_codes:
        raise ValueError('无法读取此视频。可能需要登录、链接不受支持或网络受限；请上传你已有的视频文件。')
    return out


def validate_url(url):
    url = (url or '').strip()
    try:
        parts = urlsplit(url)
        valid = parts.scheme in ('http', 'https') and parts.hostname and not parts.username and not parts.password
    except ValueError:
        valid = False
    if not valid or len(url) > 4096:
        raise ValueError('请输入完整的 http / https 视频链接。')
    return url


class VideoService:
    def __init__(self, root='data/video-imports', transcriber=None):
        self.root = Path(root).resolve()
        self.transcriber = transcriber or Transcriber()

    def load(self, url=None, upload=None):
        if not shutil.which('ffmpeg'):
            raise ValueError('请先安装 FFmpeg。')
        if upload:
            source = Path(upload)
            if not source.is_file() or source.stat().st_size > MAX_BYTES:
                raise ValueError('本地文件需小于 300 MB。')
        else:
            url = validate_url(url)
        folder = self.root / uuid.uuid4().hex
        folder.mkdir(parents=True)
        try:
            if upload:
                title = source.name
            else:
                # Ignore machine-wide downloader configuration; no browser cookies or login extraction.
                runtime = ['--js-runtimes', 'node:' + shutil.which('node')] if shutil.which('node') else []
                command([sys.executable, '-m', 'yt_dlp', *runtime, '--ignore-config', '--no-playlist', '--max-downloads', '1',
                         '--no-progress', '--no-warnings', '--socket-timeout', '20', '--retries', '1',
                         '--fragment-retries', '1', '--max-filesize', str(MAX_BYTES),
                         '--match-filter', '!is_live & duration <=? 1800',
                         '-f', 'bestaudio/best', '--write-info-json',
                         '-o', str(folder / 'source.%(ext)s'), '--', url], accepted_codes=(0, 101))
                candidates = [p for p in folder.glob('source.*') if p.suffix not in ('.json', '.part', '.ytdl')]
                if len(candidates) != 1 or candidates[0].stat().st_size > MAX_BYTES:
                    raise ValueError('未取得单个有效音频。请使用 30 分钟以内、非直播的视频，或上传本地文件。')
                source = candidates[0]
                info_file = folder / 'source.info.json'
                info = json.loads(info_file.read_text()) if info_file.exists() else {}
                if info.get('playlist_id'):
                    raise ValueError('请提供单个视频链接，不支持播放列表。')
                title = str(info.get('title') or '视频音频')[:300]
            audio = folder / 'audio.wav'
            command([shutil.which('ffmpeg'), '-nostdin', '-v', 'error', '-y', '-i', str(source.resolve()),
                     '-map', '0:a:0', '-vn', '-t', str(MAX_SECONDS + 1), '-ac', '1', '-ar', '48000',
                     '-c:a', 'pcm_s16le', str(audio)])
            duration = sf.info(audio).duration
            if not 1 <= duration <= MAX_SECONDS:
                raise ValueError('视频需为 1 秒至 30 分钟；超长内容不会被悄悄截断后接受。')
            record = dict(id=folder.name, title=title, url=url if not upload else None,
                          audio=str(audio), duration=duration, chunks=[])
            (folder / 'source.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
            # Retain normalized audio and minimal provenance, not cookies/signed media metadata.
            for p in folder.glob('source.*'):
                if p.name != 'source.json':
                    p.unlink()
            return record
        except Exception:
            shutil.rmtree(folder)
            raise

    def transcribe(self, record):
        if not record:
            raise ValueError('请先读取视频音频。')
        # Whisper receives a 16 kHz copy; the extraction source stays at 48 kHz.
        audio = Path(record['audio'])
        asr_audio = audio.with_name('asr.wav')
        command([shutil.which('ffmpeg'), '-nostdin', '-v', 'error', '-y', '-i', str(audio),
                 '-ar', '16000', str(asr_audio)])
        try:
            result = self.transcriber(str(asr_audio))
        finally:
            asr_audio.unlink(missing_ok=True)
        chunks = []
        for chunk in result.get('chunks', []):
            start, end = chunk.get('timestamp', (None, None))
            if start is None:
                continue
            end = record['duration'] if end is None else end
            start, end = float(start), float(end)
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            start, end = max(0, start), min(record['duration'], end)
            if end > start:
                chunks.append(dict(start=start, end=end, text=str(chunk.get('text', '')).strip()))
        updated = dict(record, chunks=chunks, transcript=result.get('text', ''))
        audio.with_name('source.json').write_text(json.dumps(updated, ensure_ascii=False, indent=2))
        return updated

    def clip(self, record, start, end):
        if not record:
            raise ValueError('请先读取视频音频。')
        start, end = float(start), float(end)
        if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end <= record['duration']:
            raise ValueError('起止时间必须在视频范围内，结束时间应晚于开始时间。')
        if not 1 <= end - start <= 180:
            raise ValueError('请选择 1–180 秒；声音克隆请选择 3–12 秒的完整句子。')
        target = Path(record['audio']).parent / f'clip-{uuid.uuid4().hex}.wav'
        with sf.SoundFile(record['audio']) as src:
            first, last = round(start * src.samplerate), round(end * src.samplerate)
            src.seek(first)
            data = src.read(last - first, dtype='float32')
            sf.write(target, data, src.samplerate, subtype='PCM_16')
        text = ' '.join(c['text'] for c in record['chunks'] if c['end'] > start and c['start'] < end)
        clip = dict(path=str(target), source_id=record['id'], start=start, end=end, text=text)
        target.with_suffix('.json').write_text(json.dumps(clip, ensure_ascii=False, indent=2))
        return clip


def validate_clip(record, clip, start, end, for_voice=False):
    if not record or not clip or clip['source_id'] != record['id'] or clip['start'] != float(start) or clip['end'] != float(end):
        raise ValueError('选段已改变，请重新点击「截取并试听」。')
    if for_voice and not 3 <= clip['end'] - clip['start'] <= 12:
        raise ValueError('声音样本需要 3–12 秒，请调整起止时间。')
    return clip['path']
