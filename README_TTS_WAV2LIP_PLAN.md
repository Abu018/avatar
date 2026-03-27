# TTS + Lip Sync Integration Plan (HeyGen-Style)

This document describes where to implement a **text-to-avatar video** workflow in this project:

1. Text input
2. TTS generates WAV
3. Generate talking video (Hunyuan mode) OR lip-sync existing video (Wav2Lip mode)

---

## Goal

Support two production modes behind one API/UI:

- **Hunyuan mode**: `Text -> TTS WAV -> HunyuanVideo-Avatar generation`
  - Best quality and better expression/motion coupling to audio
- **Wav2Lip mode**: `Text -> TTS WAV -> Wav2Lip(base video + audio)`
  - Faster for reusable base videos, mostly mouth-region lip sync

---

## Existing Files and Responsibilities

- `hymm_gradio/flask_audio.py`
  - Main backend API orchestration (`/predict2`)
  - Best place to route TTS + generation/lip-sync pipeline modes

- `hymm_gradio/tool_for_end2end.py`
  - Request decoding, temp file save/load, output packaging
  - Best place to parse new request fields (`text_input`, `pipeline_mode`, etc.)

- `hymm_gradio/gradio_audio.py`
  - Front-end Gradio controls and request payload assembly
  - Best place to add text box, mode select, voice settings

- `hymm_sp/sample_inference_audio.py`
  - Core audio-driven Hunyuan sampling logic
  - Keep mostly unchanged for MVP; consume generated WAV from TTS

---

## New Files to Add

### 1) `hymm_gradio/tts_service.py`

Purpose:
- Encapsulate all TTS provider logic in one place.
- Return a local `wav_path` for downstream pipeline.

Proposed API:
- `synthesize_to_wav(text: str, voice_id: str, provider: str, speed: float, language: str) -> str`

Notes:
- Start with one provider first (fastest path).
- Add retries, timeout, and clear error messages.

### 2) `hymm_gradio/wav2lip_service.py`

Purpose:
- Encapsulate Wav2Lip subprocess execution.
- Input: base face video + WAV
- Output: final lip-synced video path

Proposed API:
- `run_wav2lip(face_video_path: str, audio_path: str, output_path: str | None = None) -> str`

Notes:
- Validate ffmpeg and Wav2Lip checkpoint availability at startup.
- Surface stderr on failure for easier debugging.

---

## File-by-File Implementation Plan

## Phase 1 - TTS + Hunyuan (MVP)

### A. Update `hymm_gradio/tool_for_end2end.py`

Add support for new request keys in `process_input_dict(...)`:
- `text_input` (string)
- `pipeline_mode` (default `"hunyuan"`)
- `tts_provider` (default provider)
- `voice_id`, `tts_speed`, `tts_language`
- keep current `audio_buffer` path support

Add temporary file cleanup support:
- Ensure TTS-generated WAV is cleaned after request completion.

### B. Add `hymm_gradio/tts_service.py`

Implement:
- `synthesize_to_wav(...)`
- Output under `./temp` with UUID naming

### C. Update `hymm_gradio/flask_audio.py`

In `predict_wrap(...)`:
1. Decode request as currently done.
2. If `audio_path` missing and `text_input` exists:
   - Call TTS service to generate WAV.
   - Assign generated WAV to `driving_audio_path`.
3. Continue current Hunyuan path unchanged:
   - `data_preprocess_server(...)`
   - `generate_image_parallel(...)`
4. Return final video as current base64 response format.

Validation logic:
- If neither `audio_buffer` nor `text_input` is provided, return clear error.
- If mode is `hunyuan`, require `image_buffer`.

### D. Update `hymm_gradio/gradio_audio.py`

Add UI controls:
- `Text input` (script)
- `TTS provider` dropdown
- `Voice` dropdown/text
- `Pipeline mode` radio (`hunyuan`, later `wav2lip`)

Behavior:
- User can upload audio OR type text.
- If both are supplied, uploaded audio has priority (recommended).

---

## Phase 2 - Wav2Lip Mode

### A. Add `hymm_gradio/wav2lip_service.py`

Implement wrapper around Wav2Lip inference command:
- Validate command path and required model checkpoint.
- Return output mp4 path.

### B. Update `hymm_gradio/tool_for_end2end.py`

Parse new optional input:
- `base_video_buffer` -> `base_video_path`

### C. Update `hymm_gradio/flask_audio.py`

Add mode branch:
- `pipeline_mode == "hunyuan"` -> existing generation path
- `pipeline_mode == "wav2lip"` -> call Wav2Lip wrapper

Validation:
- `wav2lip` mode requires `base_video_buffer` (or `base_video_path`).
- If text provided and audio missing, auto-generate WAV via TTS.

### D. Update `hymm_gradio/gradio_audio.py`

For `wav2lip` mode:
- Show upload for base video
- Hide image upload as optional/unused

---

## Unified Request/Response Contract

Request (proposed):
- `image_buffer` (base64, optional in wav2lip mode)
- `audio_buffer` (base64, optional if text_input provided)
- `text_input` (optional if audio_buffer provided)
- `pipeline_mode` (`"hunyuan"` or `"wav2lip"`)
- `base_video_buffer` (required for wav2lip mode)
- `tts_provider`, `voice_id`, `tts_speed`, `tts_language`
- `text` (existing visual prompt, optional)
- `save_fps`

Response:
- Keep current structure:
  - `errCode`
  - `info`
  - `content[0].buffer` (base64 mp4)

---

## Validation Rules

- If no `audio_buffer` and no `text_input` -> fail with actionable error.
- If `pipeline_mode=hunyuan` and no `image_buffer` -> fail.
- If `pipeline_mode=wav2lip` and no `base_video_buffer` -> fail.
- If TTS fails -> include provider error details in `info`.

---

## Suggested Milestones

1. **MVP text-to-video (Hunyuan mode)**  
   `text_input -> TTS WAV -> existing Hunyuan pipeline`

2. **Add wav2lip mode**  
   `text_input -> TTS WAV -> Wav2Lip(base video)`

3. **Harden for production**  
   retries, timeouts, queueing, caching by `(text, voice_id)`, logging

---

## Estimated Build Time

- Phase 1 (TTS + Hunyuan): **4-8 hours**
- Phase 2 (Wav2Lip mode): **4-10 hours**
- Total first stable version: **1-2 days**

---

## Practical Notes

- Hunyuan mode gives better overall realism than pure Wav2Lip because it is audio-driven end-to-end.
- Wav2Lip mode is useful for high-throughput reuse of the same base video template.
- Keep all provider-specific secrets in environment variables, not in source code.

