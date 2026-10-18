# Initial validation

Validated on October 18, 2026 on Apple Silicon with Python 3.12.14.

- `pytest -q`: **15 passed**. Covers UI construction, invalid requests before model loading, duration/silence checks, output and provenance persistence, failed-run cleanup, and model reuse.
- `pip check`: no broken requirements.
- Local app launched at `127.0.0.1:7860`; browser inspection confirmed upload/record controls, transcript, sentence examples, settings, generation, audio output, and manifest download controls render.
- Real F5-TTS inference completed using the upstream project's `basic_ref_en.wav` sample and its supplied transcript. Generated text: “This is a test of the voice generation system.” Output was a nonempty, finite 24 kHz WAV. Generation took 24.22 seconds, excluding model loading/downloads. This is one smoke test, not a performance benchmark.
- Smoke-test recordings and model weights are not committed.

Tested versions: f5-tts 1.1.17, Gradio 6.30.0, torch 2.14.1, torchaudio 2.11.0, soundfile 0.14.0. Other platforms have not been tested.

Remaining: evaluate a consenting target speaker on unseen sentences; assess pronunciation, naturalness, and timbre resemblance with human listeners. No claim of measured speaker similarity or live-conversation performance is made. Microphone permission and browser recording have not been manually tested; upload is the recommended first experiment.

## M4A support

FFmpeg fallback decodes compressed reference audio to temporary PCM before existing quality checks. All 19 tests pass, including actual AAC-in-M4A decoding and passage through generation with a test model, rejection of overlong M4A, missing FFmpeg, and corrupt files. No additional real-model inference was needed for this input-format change.

## Saved profiles and spoken chat

- `pytest -q`: **28 passed**, including persisted profiles, invalid profile IDs, permission requirements, sentence buffering, cancellation isolation between sessions, suppression of audio after cancellation, and chat errors.
- Installed Ollama locally and downloaded `qwen2.5:1.5b` for the integration smoke test.
- Through the running Gradio API, saved the official F5 example as a named test voice, started a turn, received streamed text from the real local chat model, and generated three playable audio chunks with the real F5 model. First audio was ready after **23.66 seconds** including startup. Total speech synthesis across the three phrases was **36.48 seconds**. These are single-run server timings, not browser playback timings.
- Browser inspection confirmed the new controls; a real Send followed by Stop displayed the stopped status with the audio output cleared. Automated cancellation tests verify that an in-flight result is not emitted after cancellation. Audible timing/race conditions under network delay are not exhaustively tested.
- The **two-second target is not met**. Model/hardware latency and gaps between phrases remain limitations. Ten-reply target-speaker listening evaluation still requires the user's chosen reference voice and listeners. This is an operational prototype, not completed realtime performance validation.

## Long-form narration

- `pytest -q`: **35 passed**. New coverage includes Chinese/English text preservation during splitting, length limits, frozen voice settings, invalid job IDs, complete WAV/MP3 assembly, failure recovery without repeating completed segments, and WAV availability after MP3 export failure.
- Real-model Gradio API smoke test used the official F5 example voice and a two-paragraph English script. Both segments generated successfully; the combined WAV and MP3 were approximately **9.6 seconds** long.
- Reopened that task through the browser using its job ID. The resume action reused completed segments and displayed the full audio player plus WAV, MP3, and JSON download links.
- Full 10,000-character synthesis and human listening evaluation of long-form voice consistency have not been performed. The 10,000-character limit is an input cap, not a quality guarantee. Narration has no cancellation control in this version.
