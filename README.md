# VoiceMimic

A local workbench for the first milestone: generate **new sentences in a consenting speaker's vocal timbre**, then compare the reference and output. Personality, live conversation, and custom training are later milestones.

## Run locally

Use Python **3.10–3.12** (3.11 recommended), FFmpeg, and enough disk space for PyTorch and several GB of model weights. A GPU is preferable; CPU inference can be slow. F5 chooses the available device. Hardware support and inference time depend on your PyTorch installation.

On macOS, install prerequisites with Homebrew if needed:

```sh
brew install python@3.11 ffmpeg
git clone https://github.com/Cheng050504/voicemimic.git
cd voicemimic
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[voice]'
voicemimic
```

Open http://127.0.0.1:7860. On Linux, install FFmpeg with your package manager; for NVIDIA hardware, install the appropriate matching torch/torchaudio wheels for your CUDA environment before the voice extra. See [F5 installation guidance](https://github.com/SWivid/F5-TTS#installation).

The first generation downloads weights from Hugging Face and can take substantially longer. The UI starts without loading weights. There is no API key or paid service. Missing dependencies and inference failures are displayed in the status box; unexpected errors are logged in the terminal.

## Finding your way around

The interface has four focused tabs:

- **对话采集**: create a participant profile, record an answer, review its transcript, then confirm evidence-backed observations. Saving a recording opens the transcript step.
- **我的声音**: add a short reference, try a new sentence, and save the voice. Experimental spoken chat is collapsed below.
- **配音与试听**: select a saved voice, paste a script, and download the complete audio. Resume and segment editing are optional collapsed sections.
- **视频选段**: import a video link or local file, select timestamped speech, and send an excerpt to voice or interview capture.

Model settings, source metadata, and deletion controls stay collapsed until needed. Interview recordings and saved synthesis voices are currently separate workflows; an interview does not automatically create a synthesis voice.

## First experiment

1. Use your own voice or obtain the speaker's permission.
2. Upload WAV/FLAC/M4A or record **3–12 seconds** of clean speech. Finish the sentence and leave a short pause. Longer recordings must be excerpted manually with a matching transcript: the app never silently trims them.
3. Enter the exact transcript, including spoken fillers. This avoids loading an additional transcription model.
4. Enter a short new sentence or choose one of the examples. Start at speed 1.0 and seed 42.
5. Generate, listen to the reference and output, and download the WAV and JSON experiment record.

The baseline weights target English and Mandarin. Other languages and accents need explicit evaluation. The app checks duration, sample rate, silence, and likely clipping; it does **not** automatically detect background music, multiple speakers, or transcript accuracy.

## Evaluation

Hold out separate recordings and at least ten sentences from the reference material. For each output, ask the speaker and consenting familiar listeners to score **timbre resemblance**, **naturalness**, and **pronunciation** from 1–5. Track the run ID, text, seed, reference clip, and comments. Compare different references using the same text and settings. Scores are subjective; there is no claimed identity accuracy percentage.

Success for this milestone is recognizable timbre across unseen sentences, not just a replay of reference content. The app's tests use synthetic audio and a fake model to verify integration logic; passing tests does not establish cloning quality. Real-speaker listening evaluation remains necessary.

## Storage and privacy

Runs are stored in `data/runs/<id>/`: a mono reference copy, `generated-ai.wav`, and `run.json` containing transcripts, generation parameters, audio checks, timing, and a reference hash. These are excluded from Git. Set `VOICEMIMIC_OUTPUT_DIR` to change the run directory. Delete run folders to remove persisted experiments.

Gradio also keeps temporary uploads/outputs in its cache (scheduled cleanup after 24 hours while the app is running). To clear those immediately after stopping the app, remove its Gradio temporary directory; set `GRADIO_TEMP_DIR` to a dedicated folder if you want direct control. Model weights live in the Hugging Face cache separately. Inference is local, although initial dependency and weight downloads require network access. Gradio analytics is disabled by default.

The server binds only to localhost and does not create a public share link. This is a single-user prototype with no authentication. The permission checkbox is an attestation, not identity verification. Generated files are labeled as AI in names and metadata; there is no embedded watermark. Do not expose this development app as a public service.

## Development

```sh
python -m pip install -e '.[dev]'
pytest -q
```

Tests do not download model weights. The voice integration lazily imports `f5_tts.api.F5TTS`, serializes model access (including random seeds), and saves only successful runs. The UI clears previous output when starting a new attempt. No synthetic stand-in is presented as a real clone.

## Model licensing and research

This project integrates F5-TTS rather than training a model. **Upstream code is MIT; pretrained weights are CC-BY-NC**, so this baseline is for noncommercial research. A commercial deployment needs a model with suitable rights. This repository does not redistribute model weights or speaker recordings.

- [F5-TTS paper](https://arxiv.org/abs/2410.06885)
- [Official implementation and licensing](https://github.com/SWivid/F5-TTS)
- [Reference audio guidance](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/README.md)

Next milestones: speaker evaluation dataset → improve delivery/prosody → learn phrasing from approved conversations → real-time interaction.

## Milestone 02: saved voices and spoken chat

The 我的声音 tab saves a named voice under `data/voices/<voice_id>/` (reference WAV, transcript, settings, permission attestation). Save the reference once, select the saved voice, and reuse it in chat. `VOICEMIMIC_PROFILE_DIR` overrides the profile location. Profile folders can be copied together to another installation or deleted while the app is stopped; never commit recordings. Loading a profile restores its original permission attestation.

The default conversational backend is **Ollama on localhost:11434**. Install Ollama from its official site or your package manager, then run:

```sh
ollama serve
# In a second terminal; this downloads a local model:
ollama pull qwen2.5:1.5b
# In the activated VoiceMimic environment:
VOICEMIMIC_CHAT_MODEL=qwen2.5:1.5b voicemimic
```

You can enter another installed model name in Chat connection. Use local models for local-only processing; Ollama's cloud-backed models, if enabled in your Ollama installation, have different data handling. To disable that feature start Ollama with `OLLAMA_NO_CLOUD=1`. This prototype has no OpenAI API integration and requires no cloud API key. The small example chat model is a functional baseline, not a recommendation for demanding reasoning tasks.

Type a message and select **Send & speak**. A bounded background queue receives chat text while F5-TTS synthesizes completed sentences or bounded phrases. The voice model stays loaded across requests. Audio is emitted phrase by phrase, not token by token, so gaps remain possible. Browser autoplay restrictions may require pressing Play.

**Stop reply** clears buffered playback and invalidates the current turn. Sending another message also invalidates the old turn. A running F5 calculation is not forcibly interrupted; its result is discarded, and a new voice request waits for the model lock. Cancellation does not immediately reclaim GPU time. Chat history includes completed replies only, with up to the last 20 messages sent as model context. Interrupted or failed replies are not committed to history. Speech run folders still contain generated text/audio.

Timing reports measure server time until the first speech file is ready, including chat/model startup; they do not measure browser buffering or audible playback. The two-second target is aspirational and not guaranteed. Neither speaker identity consistency across ten replies nor seamless realtime playback is certified by unit tests.

## Long-form narration

The **长文案配音 · 完整音频** section accepts up to 10,000 characters and a saved voice. Click Refresh voices after saving a new voice. Generation splits text into bounded segments (up to 240 characters), preferring sentence endings; blank lines mark paragraph boundaries. All segments use a snapshot of the same reference, transcript, speed, and seed. It adds 250 ms between within-paragraph segments and 650 ms between paragraphs, in addition to any silence the model generates. No words are deliberately omitted or rewritten, but synthesized speech still needs listening review for omissions, repetitions, and voice drift.

Progress is displayed per segment. On completion, preview the full audio and download WAV, MP3, and the JSON job record. Files are local under `data/narrations/<job_id>/` (override with `VOICEMIMIC_NARRATION_DIR`). Reference audio and full script are included there and excluded from Git.

Save the displayed job ID to resume after refreshing the page or restarting the app. **继续 / 重试原任务** resumes the original script and voice snapshot, reusing completed audio segments. Changing the script or voice requires **生成完整配音**, which creates a new job. Failed MP3 export retains the WAV; retrying exports again without resynthesizing completed segments. The refinement section supports segment editing and versions; narration has no cancel button, and jobs run serially and may take minutes on this hardware. Stop reply controls chat only.


## Narration refinement (last narration milestone)

Under **配音精修**, load an existing narration job and choose a segment. The original script is read-only. Edit **实际读法** for names, numbers, or abbreviations; set the segment speed, extra pause (0–5 seconds), and random seed; then generate only that segment. A new seed permits variation but does not guarantee a better result.

Successful regeneration archives the original and adds a new version. The latest version is adopted automatically. Select a historical version to audition it, then click **采用此版本** to restore its audio and generation settings. **只保存停顿** changes the extra silence without speech inference; it does not trim silence already produced by the model. Selecting a historical version restores its saved pause; adjust again if desired.

After any edit, the UI clears previous full-output downloads. Click **重新合并并导出完整音频** to rebuild WAV/MP3 from the chosen segment versions without regenerating other completed segments. Failed regeneration preserves the current audio and job metadata. Version files remain inside the job folder. Already downloaded exports and views open in other browser tabs remain older snapshots; re-export before using them.

This finishes the planned narration tooling scope. Subsequent development returns to consented conversation collection, voice/prosody modeling, and evidence-based personality/expression modeling; see ROADMAP.md.

## Guided conversation capture and reviewed personal profiles

The **对话采集与审核** section returns to the main goal. Create a participant profile with consent, choose/edit an interview question, and record only the participant's answer (1–180 seconds, up to 100 MB). Eight prompts cover stories, explanations, changing opinions, disagreement, trade-offs, support, humor, and misunderstandings. A first 10–15 minute interview is a collection target, not enough evidence to claim personality replication. This version is turn-by-turn recording/upload, not live duplex interviewing or automatic speaker separation.

Each sample stores its question, context, participant attribution, capture time, normalized audio, original ASR output/timestamps, corrected transcript, and review status. Audio is saved before transcription, so ASR failure does not lose the recording. Click automatic transcription or type a transcript yourself. Transcription uses local `openai/whisper-base` through Transformers; first use downloads the model. The existing `[voice]` extra includes the dependencies. English and Mandarin need participant review, especially with noise, names, or accents. ASR chunk timestamps refer to the original transcription, not automatically realigned corrected text.

Review the sample and check approval before generating observations. The locally running Ollama model proposes up to six observations from reviewed transcripts only (20,000-character analysis cap). Each proposal must cite a real sample and an exact quote, otherwise it is discarded. Exact quote matching verifies provenance, **not whether the interpretation is true**. All observations start unapproved; participants can edit claims, correct explicit-self-report versus inference classification, and approve individual entries. Source controls open the original answer for review. This is not psychometric diagnosis, training, or an automatic change to the saved voice.

Export includes only approved samples and approved observations with valid sources, plus evidence coverage counts. Editing/revoking a sample invalidates its observations; deleting a sample removes its saved audio, transcript, and derived observations. Exported profile snapshots in that folder are invalidated on mutations. Data lives under `data/interviews` (override `VOICEMIMIC_INTERVIEW_DIR`) and is excluded from Git. User-downloaded copies and Gradio/browser caches are separate; deletion does not retract them. Do not record multiple speakers as one participant: this version does not verify identity or detect overlapping voices.

## Optional recording noise reduction

Both **对话采集** and **我的声音** include a collapsed **录音有杂音？先试听降噪** panel. Preview the original and processed audio, then explicitly choose **使用降噪版** to use it for subsequent saving or synthesis. **恢复原音** restores the decoded original. Changing the uploaded recording invalidates adoption of an unrelated preview. Adjusting strength after adoption starts from the original rather than stacking filters.

The lightweight local baseline uses FFmpeg's [afftdn filter](https://www.ffmpeg.org/ffmpeg-filters.html#afftdn), with 6 dB (light) or 10 dB (standard) reduction, noise tracking, and smoothed gains. It applies no pitch/time changes, hard gate, EQ, or loudness normalization. Sample rate, channel count, and frame count are checked. Spectral filtering can still alter breathiness and consonants; use light mode first and compare by listening. Competing voices, music, clipping, and room echo are not reliably repaired by this baseline.

WAV/FLAC/M4A decoding uses FFmpeg. Limits are 100 MB, 16–96 kHz, mono/stereo, and 1–180 seconds for interviews or 1–12 seconds for previewing a voice reference; synthesis still requires 3–12 seconds. Overlong inputs are rejected, never silently accepted as excerpts. Original source bytes, decoded originals, processed WAVs, and processing metadata persist under `data/denoise/<id>/` (override with `VOICEMIMIC_DENOISE_DIR`). These files are excluded from Git and are not removed when an interview or voice profile is deleted; remove the corresponding denoise folder separately if needed.

## Video links and transcript-based excerpts

Open **视频选段**, paste a link, and choose **读取链接并识别**. The app uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) to retrieve accessible media, FFmpeg to extract audio, and the existing local Whisper model to generate timestamped text. Select one or more sentences to fill a continuous start/end interval, or enter seconds manually; then **截取并试听**. Multiple selected sentences include all intervening audio. Review the suggested text against playback, especially after cutting within a sentence.

Use **用作声音样本** to fill the reference recording and transcript in 我的声音 (3–12 seconds), or **送到对话采集** to fill a recording and its source context (1–180 seconds). Existing permission and review steps still apply. Changing the source or clip boundaries requires a fresh excerpt before adoption. Denoising remains available after importing the excerpt.

Limits: one non-live video, at most 30 minutes, 300 MB download/file limit, and a four-minute retrieval timeout. Downloads with unknown lengths/sizes are also checked after retrieval; FFmpeg decodes only enough to reject overlong media. Audio is normalized to mono 48 kHz PCM for editing; Whisper receives a temporary 16 kHz copy. This does not improve source fidelity. Word timing and speaker separation are not guaranteed; this version does not isolate speakers or remove background music. The existing Whisper chunked transcription can be inaccurate, especially for long/noisy material.

Link support depends on platform changes, network restrictions, and the extractor. No login, browser-cookie extraction, paywall/DRM bypass, or playlist import is provided. A local video/audio upload fallback is included. If recognition fails after retrieval, the imported audio remains available for manual clipping or transcription retry. For full YouTube extractor support, yt-dlp may require an installed supported JavaScript runtime; the app enables Node when it is on PATH, and yt-dlp's default Deno support remains available. Installing the project includes yt-dlp's default dependencies.

Audio, transcript timestamps, source URL/title, and excerpt provenance are saved under `data/video-imports/<id>/` (override `VOICEMIMIC_VIDEO_DIR`). Downloaded container files and downloader metadata are removed after successful conversion. Import folders are separate from voice/interview records and must be deleted separately. This remains a localhost-only, single-user prototype; do not expose the URL-fetching interface as a public service.
