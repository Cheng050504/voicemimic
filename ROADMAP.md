# Main objective: learn a person's voice and conversational behavior through dialogue

Narration refinement is the final planned narration-tool milestone. Do not keep expanding into a general audio editor unless requested. The core goal is a correctable approximation of the consenting speaker's voice, delivery, and characteristic responses.

## Next: conversation capture and review

Record participant-authorized conversations with speaker attribution, timestamped audio, transcripts, and context. Preserve pauses and corrections. Let the participant review, remove, and approve samples. Separate the participant's speech from the interviewer and generated assistant speech. Begin with uploads/review before assuming reliable realtime turn-taking.

Acceptance: one reviewed conversation can be split into speaker-specific, timestamped samples and linked transcript excerpts, with deletable source data and explicit provenance.

## Voice and prosody from multiple conversations

Collect examples across explaining, storytelling, casual chat, questions, and disagreement. Compare reference selection and conditioning across these contexts, measuring timbre, speech rate, pitch movement, emphasis, and pauses separately. Do not assume pitch reveals personality or that a universal speaker embedding transfers between models.

Acceptance: participant/listener comparison on held-out utterances and contexts; report errors and sample coverage, not an unsupported identity accuracy score.

## Expression and personality approximation

Build an editable profile backed by approved conversation excerpts: vocabulary, sentence structure, humor, response habits, and explicitly stated preferences. Keep facts separate from inferences, mark uncertainty, and allow corrections. Do not turn generated guesses into source evidence.

Acceptance: compare generic, profile-conditioned, and example-retrieval responses on unseen questions. Ask the participant whether the words and reactions fit; test consistency across extended conversations.

## Integrated conversation

Combine reviewed response style with context-appropriate voice delivery. Revisit speech-model latency and interruption handling with measured end-to-end playback latency. Current F5-TTS performance has not met the two-second target; do not describe this prototype as natural realtime conversation.
