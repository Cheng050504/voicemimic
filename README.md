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

## First experiment

1. Use your own voice or obtain the speaker's permission.
2. Upload WAV/FLAC/M4A or record **3–12 seconds** of clean speech. Finish the sentence and leave a short pause. Longer recordings must be excerpted manually with a matching transcript: the app never silently trims them.
3. Enter the exact transcript, including spoken fillers. This avoids loading an additional transcription model.
4. Enter a short new sentence or choose one of the ten examples. Start at speed 1.0 and seed 42.
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

The new section below the timbre lab saves a named voice under `data/voices/<voice_id>/` (reference WAV, transcript, settings, permission attestation). Save the reference once, select the saved voice, and reuse it in chat. `VOICEMIMIC_PROFILE_DIR` overrides the profile location. Profile folders can be copied together to another installation or deleted while the app is stopped; never commit recordings. Loading a profile restores its original permission attestation.

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

Save the displayed job ID to resume after refreshing the page or restarting the app. **继续 / 重试原任务** resumes the original script and voice snapshot, reusing completed audio segments. Changing the script or voice requires **生成完整配音**, which creates a new job. Failed MP3 export retains the WAV; retrying exports again without resynthesizing completed segments. This version does not offer segment editing or a cancel button for narration; jobs run serially and may take minutes on this hardware. Stop reply controls chat only.
