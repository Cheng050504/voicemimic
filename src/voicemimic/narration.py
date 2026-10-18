"""Resumable long-form narration with frozen voice settings and file-based assembly."""
import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf


def split_script(text, limit=240):
    text = (text or "").strip()
    if not text or len(text) > 10000:
        raise ValueError("请输入 1–10,000 个字符的文案。")
    chunks = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        while paragraph:
            if len(paragraph) <= limit:
                end = len(paragraph)
            else:
                boundaries = list(re.finditer(r'[。！？.!?；;](?:[”’"\']|\s)*', paragraph[:limit]))
                end = boundaries[-1].end() if boundaries else paragraph.rfind(" ", 0, limit)
                if end < 1:
                    end = limit
            chunk, paragraph = paragraph[:end].strip(), paragraph[end:].strip()
            if chunk:
                chunks.append({"text": chunk, "pause": 0.25 if paragraph else 0.65})
    chunks[-1]["pause"] = 0
    return chunks


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class NarrationService:
    def __init__(self, voice_service, root="data/narrations"):
        self.voice_service = voice_service
        self.root = Path(root)

    def create(self, text, profile):
        chunks = split_script(text)
        if not profile.get("permission_confirmed"):
            raise ValueError("请使用已确认授权的声音。")
        job_id = uuid.uuid4().hex
        folder = self.root / job_id
        folder.mkdir(parents=True)
        shutil.copyfile(profile["reference"], folder / "reference.wav")
        job = {"id": job_id, "ai_generated": True, "script": text.strip(),
               "voice_name": profile["name"], "voice_id": profile["id"],
               "transcript": profile["transcript"], "speed": profile["speed"], "seed": profile["seed"],
               "chunks": chunks, "status": "pending", "error": None}
        save_json(folder / "job.json", job)
        return job_id

    def load(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("请先创建配音任务，或输入已保存的任务编号。")
        folder = self.root / job_id
        try:
            job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("找不到此配音任务。") from exc
        return folder, job

    def run(self, job_id):
        folder, job = self.load(job_id)
        manifest = folder / "job.json"
        total = len(job["chunks"])
        try:
            for i, chunk in enumerate(job["chunks"]):
                target = folder / f"part-{i:04d}.wav"
                if target.exists():
                    # A part is reusable only after an atomic successful write.
                    info = sf.info(target)
                    if info.frames > 0:
                        yield {"status": f"已恢复 {i + 1}/{total} 段", "job_id": job_id}
                        continue
                job.update(status="generating", error=None)
                save_json(manifest, job)
                yield {"status": f"正在生成第 {i + 1}/{total} 段…", "job_id": job_id}
                audio, _, _ = self.voice_service.generate(
                    str(folder / "reference.wav"), job["transcript"], chunk["text"], True,
                    job["speed"], job["seed"],
                )
                temporary = target.with_suffix(".tmp")
                shutil.copyfile(audio, temporary)
                temporary.replace(target)
                chunk["file"] = target.name
                save_json(manifest, job)
                yield {"status": f"已完成 {i + 1}/{total} 段", "job_id": job_id}
            yield {"status": "正在合并完整音频并导出 MP3…", "job_id": job_id}
            wav = folder / "narration-ai.wav"
            temp_wav = folder / "assembled.tmp.wav"
            sample_rate = sf.info(folder / "part-0000.wav").samplerate
            with sf.SoundFile(temp_wav, "w", samplerate=sample_rate, channels=1, subtype="PCM_16") as out:
                for i, chunk in enumerate(job["chunks"]):
                    with sf.SoundFile(folder / f"part-{i:04d}.wav") as source:
                        if source.samplerate != sample_rate:
                            raise ValueError("分段采样率不一致，无法合并。")
                        for block in source.blocks(blocksize=65536, dtype="float32", always_2d=True):
                            out.write(block.mean(axis=1))
                    if chunk["pause"]:
                        out.write(np.zeros(round(sample_rate * chunk["pause"]), dtype=np.float32))
            temp_wav.replace(wav)
            # WAV remains downloadable even if MP3 export fails.
            yield {"status": "WAV 已完成，正在导出 MP3…", "job_id": job_id, "wav": str(wav)}
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("WAV 已完成；导出 MP3 需要安装 FFmpeg。安装后点击继续 / 重试。")
            mp3 = folder / "narration-ai.mp3"
            temporary_mp3 = folder / "export.tmp.mp3"
            subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(wav),
                            "-codec:a", "libmp3lame", "-q:a", "2", str(temporary_mp3)],
                           check=True, capture_output=True, timeout=120)
            temporary_mp3.replace(mp3)
            job.update(status="complete", error=None, duration_seconds=sf.info(wav).duration)
            save_json(manifest, job)
            yield {"status": f"完成：{total} 段，共 {job['duration_seconds']:.1f} 秒。", "job_id": job_id,
                   "wav": str(wav), "mp3": str(mp3), "manifest": str(manifest)}
        except Exception as exc:
            job.update(status="failed", error=str(exc))
            save_json(manifest, job)
            raise
