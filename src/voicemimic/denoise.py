"""Conservative local noise reduction with immutable originals and explicit adoption."""
import hashlib
import json
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

PRESETS = {'保真优先（推荐）': 6, '轻度': 6, '轻度（推荐）': 6, '标准': 10}


def fingerprint(path):
    if not path or not Path(path).is_file():
        raise ValueError('请先上传或录制音频。')
    if Path(path).stat().st_size > 100 * 1024 * 1024:
        raise ValueError('录音不能超过 100 MB。')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_ffmpeg(args):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise ValueError('降噪需要 FFmpeg，请安装后重试。')
    try:
        subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', *args],
                       check=True, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('处理超时，请使用更短的录音。') from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError('音频处理失败，请检查文件格式及 FFmpeg 是否支持 afftdn。') from exc


def denoise(path, preset='保真优先（推荐）', max_seconds=180, root='data/denoise'):
    source_hash = fingerprint(path)
    if preset not in PRESETS:
        raise ValueError('请选择保真优先、轻度或标准降噪。')
    folder = Path(root).resolve() / uuid.uuid4().hex
    folder.mkdir(parents=True)
    try:
        source = folder / ('source' + Path(path).suffix)
        shutil.copyfile(path, source)
        original, cleaned = folder / 'original.wav', folder / 'denoised.wav'
        run_ffmpeg(['-i', str(source), '-map', '0:a:0', '-vn', '-t', str(max_seconds + 1),
                    '-c:a', 'pcm_f32le', str(original)])
        audio, sr = sf.read(original, always_2d=True, dtype='float32')
        if not 1 <= len(audio) / sr <= max_seconds:
            raise ValueError(f'请使用 1–{max_seconds} 秒的录音，降噪不会自动截短。')
        if not 16000 <= sr <= 96000 or audio.shape[1] > 2:
            raise ValueError('请使用 16–96 kHz 的单声道或双声道录音。')
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
            raise ValueError('录音含无效或超范围采样，请重新导出。')
        if float(np.sqrt(np.mean(audio ** 2))) < 0.0001:
            raise ValueError('录音太安静，无法可靠降噪。')
        # Modest attenuation, smooth gains, no EQ, gate, pitch/time change or loudness boost.
        # afftdn delays audio by two analysis hops. Flush and remove that delay
        # before blending, otherwise the dry/wet mixture cancels voice harmonics.
        delay = 2 * (sr // 80)
        filter_text = (f'apad=pad_len={delay},afftdn=nr={PRESETS[preset]}:nf=-45:tn=1:gs=6,'
                       f'atrim=start_sample={delay}:end_sample={delay + len(audio)},asetpts=PTS-STARTPTS')
        run_ffmpeg(['-i', str(original), '-af', filter_text, '-c:a', 'pcm_f32le', str(cleaned)])
        result, result_sr = sf.read(cleaned, always_2d=True, dtype='float32')
        if result_sr != sr or result.shape != audio.shape or not np.isfinite(result).all():
            raise ValueError('降噪输出校验失败，原音频未改变。')
        mix = 0.35 if preset == '保真优先（推荐）' else 1.0
        if mix < 1:
            result = (1 - mix) * audio + mix * result
            sf.write(cleaned, result, sr, subtype='FLOAT')
        record = dict(delay_compensation_samples=delay, wet_mix=mix, source_hash=source_hash, original=str(original), cleaned=str(cleaned),
                      cleaned_hash=fingerprint(cleaned), preset=preset, filter=filter_text,
                      sample_rate=sr, channels=audio.shape[1], frames=len(audio))
        (folder / 'processing.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return record
    except Exception:
        shutil.rmtree(folder)
        raise


def checked_preview(current, record):
    if not record or fingerprint(current) not in (record['source_hash'], record['cleaned_hash'], fingerprint(record['original'])):
        raise ValueError('录音已更换，请重新预览降噪，避免用错音频。')
    return record
