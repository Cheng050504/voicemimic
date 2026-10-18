# Initial validation

Validated on October 18, 2026 on Apple Silicon with Python 3.12.14.

- `pytest -q`: **15 passed**. Covers UI construction, invalid requests before model loading, duration/silence checks, output and provenance persistence, failed-run cleanup, and model reuse.
- `pip check`: no broken requirements.
- Local app launched at `127.0.0.1:7860`; browser inspection confirmed upload/record controls, transcript, sentence examples, settings, generation, audio output, and manifest download controls render.
- Real F5-TTS inference completed using the upstream project's `basic_ref_en.wav` sample and its supplied transcript. Generated text: “This is a test of the voice generation system.” Output was a nonempty, finite 24 kHz WAV. Generation took 24.22 seconds, excluding model loading/downloads. This is one smoke test, not a performance benchmark.
- Smoke-test recordings and model weights are not committed.

Tested versions: f5-tts 1.1.17, Gradio 6.30.0, torch 2.14.1, torchaudio 2.11.0, soundfile 0.14.0. Other platforms have not been tested.

Remaining: evaluate a consenting target speaker on unseen sentences; assess pronunciation, naturalness, and timbre resemblance with human listeners. No claim of measured speaker similarity or live-conversation performance is made. Microphone permission and browser recording have not been manually tested; upload is the recommended first experiment.
