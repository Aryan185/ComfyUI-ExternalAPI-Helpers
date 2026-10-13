# ComfyUI-ExternalAPI-Helpers

A collection of powerful custom nodes for ComfyUI that connect your local workflows to closed-source AI models via their APIs. Use Google's Gemini, Imagen, Veo 3.1, OpenAI's GPT-6 and GPT Image 2.5, ElevenLabs v4, and Black Forest Labs' FLUX models directly within ComfyUI.

## Key Features

*   **FLUX Kontext Pro & Max:** Image-to-image transformations using the FLUX models via the Replicate API.
*   **Flux.2 (Replicate):** Generate images using the latest FLUX.2 models (Pro, Max, Dev) via Replicate.
*   **Gemini Chat:** Google's frontier multimodal models (Gemini 3.8 Flash, 3.6 Flash, 3.5 Flash-Lite, 3.1 Pro). Analyze images, process audio waveforms, and generate responses with dynamic thinking controls.
*   **Gemini Speaker Diarization:** Separate audio into individual speaker tracks with high-precision timestamping using Gemini 3.8.
*   **GPT Image Edit:** Prompt-based image editing and inpainting with OpenAI's latest `gpt-image-2.5-sunburst` and `gpt-image-2.5-flare` alongside GPT Image 2.
*   **OpenAI LLM:** Access OpenAI's frontier reasoning and generation models (GPT-6, GPT-5.6 family, GPT-5.5, GPT-5.4) using the native Responses API with selectable reasoning effort (`none`, `low`, `medium`, `high`).
*   **OpenAI Text-to-Speech:** Generate speech with OpenAI's TTS models (`gpt-4o-mini-tts`, `tts-1`, `tts-1-hd`) featuring all 13 built-in voices and natural language tone/style instructions.
*   **ElevenLabs TTS:** Generate speech using ElevenLabs' newest **Eleven v4** and **Eleven v4 Turbo** models with automatic parameter branching, alongside legacy v3 and multilingual engines.
*   **Gemini 3.8 TTS:** Next-gen speech synthesis powered by Gemini 3.8 with access to Google's Extended Voice Library (2,000+ voices like *Bodi*, *Lumi*, *Koda*), native `speech_metadata` style direction, and inline vocal tags (`<breath>`, `<laugh>`, `<short pause>`, `<sigh>`, `<cough>`).
*   **Gemini TTS (Legacy):** Dedicated backwards-compatible node for earlier Gemini 2.5 and 3.1 celestial voice endpoints.
*   **Nano Banana:** Google's native image generation and editing using the latest **`gemini-nano-banana-2.1`**, featuring 1K/2K/4K resolution control and multi-image consistency.
*   **Veo Video Generator (Gemini API & Vertex AI):** Generate video clips using Google's **Veo 3.1** family (`veo-3.1-generate-preview`, `veo-3.1-fast`, `veo-3.1-lite` / Vertex `001` releases) with native synchronized audio, 1080p HD support, and flexible durations.
*   **Google Imagen Generator & Edit:** Create and edit images with Google's Imagen models, with support for Vertex AI.
*   **Groq LLM:** Fast text generation using Groq-hosted models, with optional image input for supported vision models.
*   **Groq Orpheus TTS:** Generate speech using Groq-hosted Orpheus text-to-speech models.
*   **Tripo Text-to-3D & Image-to-3D:** Generate and convert images into production 3D models using Tripo AI.

---

## 🚀 Installation

1.  Navigate to your ComfyUI installation directory.
2.  Go into the `custom_nodes` folder:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
3.  Clone this repository:
    ```bash
    git clone https://github.com/Aryan185/ComfyUI-ExternalAPI-Helpers.git
    ```
4.  Install the required Python packages:
    ```bash
    cd ComfyUI-ExternalAPI-Helpers
    pip install -r requirements.txt
    ```
5.  **Restart ComfyUI.** The nodes will be available in the "Add Node" menu.

---

## 🔑 Prerequisites: API Keys

*   **FLUX Nodes (Replicate):** [Replicate API Token](https://replicate.com/account/api-tokens).
*   **Google AI Studio Nodes (Gemini Chat, Diarization, Nano Banana, Gemini 3.8 TTS, Veo Gemini API):** [Google AI Studio API Key](https://aistudio.google.com/app/api-keys).
*   **OpenAI Nodes (GPT Image Edit, OpenAI LLM, OpenAI TTS):** [OpenAI API Key](https://platform.openai.com/api-keys).
*   **ElevenLabs TTS Node:** [ElevenLabs API Key](https://elevenlabs.io/).
*   **Groq Nodes (LLM, Orpheus TTS):** [Groq API Key](https://console.groq.com/keys).
*   **Tripo Nodes (Text-to-3D, Image-to-3D):** [Tripo API Key](https://tripo3d.ai/).
*   **Vertex AI Nodes (Imagen Edit, Veo Vertex AI):** Google Cloud Project ID, location, and service account JSON key.

### Using Environment Variables (.env file)

Instead of pasting raw API keys into every node, store them in a `.env` file in the node directory:

```env
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
XI_API_KEY=your_elevenlabs_key_here
GROQ_API_KEY=your_groq_key_here
REPLICATE_API_TOKEN=your_replicate_token_here
TRIPO_API_KEY=your_tripo_api_key_here
```

In any node's `api_key` field, you can simply type the variable name (e.g. `GEMINI_API_KEY`) instead of exposing your actual secret.

---

## 📚 Node Guide

### Flux Kontext Pro / Max
*   **Category:** `image/edit`
*   **Key Inputs:** `image`, `aspect_ratio` (use `match_input_image` to preserve composition)
*   **Output:** `image`

### Flux.2 (Replicate)
*   **Category:** `image/generation`
*   **Key Inputs:** `image_1` to `image_5` (optional, for image-to-image tasks)
*   **Output:** `image`

### Gemini Chat
*   **Category:** `text/generation`
*   **Models:** `gemini-3.8-flash` (default), `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-pro-preview`, dynamic aliases.
*   **Key Inputs:** `prompt`, `thinking`, `thinking_budget`, `google_search`, `system_instruction`, `image` (optional), `audio` (optional)
*   **Output:** `response` (text)

### Gemini Speaker Diarization
*   **Category:** `audio/diarise`
*   **Key Inputs:** `audio`, `num_speakers`, `model`, `thinking`, `thinking_budget`
*   **Output:** `speaker_1` to `speaker_4` (isolated audio tracks)

### GPT Image Edit
*   **Category:** `image/generation`
*   **Models:** `gpt-image-2.5-sunburst` (default), `gpt-image-2.5-flare`, `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1`
*   **Key Inputs:** `prompt`, `image_1` to `image_5` (optional reference/edit inputs), `mask` (optional, white area = inpainting region), `background`, `quality`, `size`
*   **Output:** `image`

### OpenAI LLM
*   **Category:** `text/generation`
*   **Models:** `gpt-6` (default), `gpt-6.1-sol`, `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`, `gpt-4.1`
*   **Key Inputs:** `prompt`, `reasoning_effort` (`none`, `low`, `medium`, `high`), `max_output_tokens`, `system_instruction`, `image` (optional)
*   **Output:** `response` (text)

### OpenAI Text-to-Speech
*   **Category:** `audio/generation`
*   **Models:** `gpt-4o-mini-tts` (default), `tts-1`, `tts-1-hd`
*   **Voices:** 13 voices (`alloy`, `ash`, `ballad`, `cedar`, `coral`, `echo`, `fable`, `marin`, `nova`, `onyx`, `sage`, `shimmer`, `verse`)
*   **Key Inputs:** `text`, `voice`, `response_format`, `speed`, `instructions` (natural-language tone/pacing direction on `gpt-4o-mini-tts`)
*   **Output:** `audio`

### ElevenLabs TTS
*   **Category:** `audio/generation`
*   **Models:** `eleven_v4` (default), `eleven_v4_turbo`, `eleven_v3`, `eleven_multilingual_v2`, `eleven_flash_v2_5`, `eleven_flash_v2`, `eleven_turbo_v2_5`
*   **Key Inputs:** `text`, `voice_id`, `stability`, `similarity_boost`, `speed`, `style`, `use_speaker_boost`, `previous_text` / `next_text` (automatically branches parameters to match v4 requirements)
*   **Output:** `audio`

### Gemini 3.8 TTS
*   **Category:** `audio/generation`
*   **Models:** `gemini-3.8-flash-tts` (default), `gemini-3.8-flash-lite-tts`
*   **Key Inputs:** 
    *   `text`: Supports point-in-time inline tags directly in the transcript (e.g. `<breath>`, `<laugh>`, `<short pause>`, `<long pause>`, `<sigh>`, `<cough>`, `<gasp>`, `<whispering>`).
    *   `voice`: Open string targeting any of the 2,000+ Extended Voice Library voices from AI Studio (e.g. `Bodi`, `Lumi`, `Koda`, `Nika`, `Tavi`) or legacy celestial voices (`Puck`, `Zephyr`).
    *   `style`: Global acting/delivery note passed straight to native `speech_metadata`.
*   **Output:** `audio`

### Gemini TTS (Legacy)
*   **Category:** `audio/generation`
*   **Models:** `gemini-2.5-flash-preview-tts`, `gemini-2.5-pro-preview-tts`, `gemini-3.1-flash-tts-preview`
*   **Key Inputs:** Fixed 30 celestial voices dropdown (`Puck`, `Zephyr`, etc.), `pace`, `accent`, `scene`, `audio_profile` prompt-director options.
*   **Output:** `audio`

### Nano Banana
*   **Category:** `image/generation`
*   **Models:** `gemini-nano-banana-2.1` (default), `gemini-3-pro-image`, `gemini-3.1-flash-lite-image`, `gemini-3.1-flash-image`, `gemini-2.5-flash-image`
*   **Key Inputs:** `prompt`, `aspect_ratio`, `resolution` (`1K`, `2K`, `4K`), `image_1` to `image_5` (multi-image consistency & editing), `google_search`
*   **Output:** `image`

### Veo Video Generator (Gemini API)
*   **Category:** `video/generation`
*   **Models:** `veo-3.1-generate-preview` (default), `veo-3.1-fast-generate-preview`, `veo-3.1-lite-generate-preview`, `veo-2.0-generate-001`
*   **Key Inputs:** `prompt`, `aspect_ratio` (16:9, 9:16), `resolution` (720p, 1080p on 3.1), `duration_seconds` (4 to 8s), `image` (optional, for image-to-video)
*   **Output:** `frames`

### Veo Video Generator (Vertex AI)
*   **Category:** `video/generation`
*   **Models:** `veo-3.1-generate-001` (default), `veo-3.1-fast-generate-001`, `veo-3.0-generate-001`, `veo-2.0-generate-001`
*   **Key Inputs:** `project_id`, `location`, `service_account`, `resolution`, `aspect_ratio`, `duration_seconds`, `generate_audio` (native synchronized audio on 3.x), `first_frame` / `last_frame`
*   **Output:** `frames`, `audio`

### Google Imagen Generator & Edit
*   **Category:** `image/generation` & `image/edit`
*   **Key Inputs:** Generation controls (`aspect_ratio`, `image_size`, `guidance_scale`) and Vertex AI edit modes (inpainting, outpainting, background swap).
*   **Output:** `images` / `edited_images`

### Groq LLM & Groq Orpheus TTS
*   **Category:** `text/generation` & `audio/generation`
*   **Key Inputs:** Fast inference using Groq Cloud endpoints for text/vision reasoning and Orpheus voice generation.
*   **Output:** `response` (text) / `audio`

### Tripo Text-to-3D & Image-to-3D
*   **Category:** `3D/generation`
*   **Key Inputs:** `prompt` or `image`, `model_version` (v3.0, v2.5, v2.0, Turbo), `texture_quality`, `face_limit`, `pbr`
*   **Output:** `glb` (3D model file)

---

## Acknowledgements

*   The [ComfyUI](https://github.com/comfyanonymous/ComfyUI) team for building the modular node platform.
*   [Google](https://deepmind.google/technologies/gemini/) for Gemini 3.8, Nano Banana, Imagen, and Veo 3.1.
*   [OpenAI](https://openai.com/) for GPT-6, GPT Image 2.5, and speech models.
*   [ElevenLabs](https://elevenlabs.io/) for Eleven v4 and conversational audio synthesis.
*   [Groq](https://groq.com/) for high-throughput LPU inference.
*   [Black Forest Labs](https://www.blackforestlabs.ai/) and [Replicate](https://replicate.com/) for FLUX model infrastructure.
*   [Tripo AI](https://www.tripo3d.ai/) for native text and image-to-3D APIs.