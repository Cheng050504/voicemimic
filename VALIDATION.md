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

## Narration refinement

- **40 tests passed**. New coverage verifies single-segment edits preserve the original script and other audio, archived versions can be restored, pause-only edits avoid inference, re-export reuses segments, failed regeneration preserves the old version, and pause bounds reject invalid values.
- Real F5-TTS API test regenerated only segment 1 of the existing two-paragraph official-sample narration, changing its spoken wording, speed, seed, and pause. The original was retained as version 1; the new version was adopted as version 2. Full export completed at approximately 9.5 seconds.
- Browser verification loaded the task and displayed the original and corrected text, selected parameters, current audio, historical audio, and version/adoption controls.
- A 3–5 minute human-reviewed production script has not yet been evaluated. This validates the editing workflow, not long-form perceptual quality or a person's personality similarity.

## Guided interview capture and reviewed expression profiles

- **46 tests pass**. Coverage includes consent, record/transcribe/review/export, exclusion of unreviewed samples, exact-quote grounding, separate observation approval, evidence invalidation after edits, confirmed deletion and derived-data removal, and invalid IDs.
- Real local Whisper-base inference transcribed the official F5 example recording as “Some call me nature, others call me mother nature.” and returned a 0–5 second timestamped chunk.
- Real local Ollama `qwen2.5:1.5b` produced one valid, source-grounded observation on a clearly synthetic text fixture in a temporary interview store. The observation remained unapproved. This tests the pipeline, not actual personality fidelity.
- No 10–15 minute participant interview, microphone permission flow, Mandarin transcription benchmark, or human personality evaluation has been completed. Voice/prosody extraction and training are future milestones. Exact quotes do not establish the truth of an inferred trait.

## Focused interface navigation

- **46 tests pass** after reorganizing the interface into capture, saved voice, and narration tabs. No model or storage behavior was changed.
- Restarted the local service; homepage returned HTTP 200. The live Gradio config exposes all six main/step tabs, and all callback input/output component IDs resolve.
- Saving an interview recording and opening an observation source select the transcript-review step. Export, resume, and editing controls are grouped with their relevant tasks.
- Automated browser preview was rejected by the browser security policy. This change has code/config verification, but no completed visual or interactive browser verification.

## Conservative recording denoising

- **49 tests passed**, including actual FFmpeg filtering and M4A decoding. Synthetic noisy harmonic audio verifies reduced noise-only tail energy, bounded voice-region RMS change, unchanged rate/channel/frame counts, source-file integrity, stale-preview rejection, and rejection/cleanup of silent or overlong recordings.
- These signal checks do not establish perceptual timbre preservation on real speakers. No human A/B evaluation or browser interaction test has been completed for this feature.

## Video import and timestamp selection

- **53 tests pass**, including real FFmpeg extraction from a generated MP4 and actual yt-dlp retrieval over a local HTTP fixture. Checks cover 16 kHz ASR input, timestamp fallback, exact sample-count cropping, multi-sentence text overlap, invalid ranges/URLs, stale selection rejection, voice duration limits, original-upload preservation, and failed-import cleanup.
- Real local Whisper-base recognized the bundled official F5 English example as “Some call me nature, others call me mother nature.” with a 0–5 second segment. A five-second excerpt was generated from the retained 48 kHz audio.
- Public-platform extraction, long-video transcription accuracy, browser interactions, and human listening quality were not verified. Platform availability is not guaranteed by the local HTTP fixture test.
- Running Gradio API smoke test completed real Whisper import, five-second excerpt creation, reference audio/transcript adoption, interview audio/source-context adoption, and rejection of an excerpt after its requested boundary changed. Homepage returned HTTP 200 and all callback component references resolved. This is API verification, not a browser visual test.
