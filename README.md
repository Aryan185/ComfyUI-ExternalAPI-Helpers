# ComfyUI-ExternalAPI-Helpers

A collection of powerful custom nodes for ComfyUI that connect your local workflows to closed-source AI models via their APIs. Use Google's Gemini, Imagen, Veo, OpenAI's GPT-Image-1, and Black Forest Labs' FLUX models directly within ComfyUI.

## Key Features

*   **FLUX Kontext Pro & Max:** Image-to-image transformations using the FLUX models via the Replicate API.
*   **Flux.2 (Replicate):** Generate images using the latest FLUX.2 models (Pro, Max, Dev) via Replicate.
*   **Gemini Chat:** Google's powerful multimodal AI. Ask questions about an image, generate detailed descriptions or create prompts for other models. Supports thinking budget controls for applicable models. Now supports audio input.
*   **Gemini Segmentation:** Generate segmentation masks for objects in an image using Gemini.
*   **Gemini Speaker Diarization:** Separate audio into different speaker tracks using Gemini.
*   **GPT Image Edit:** OpenAI's `gpt-image-1` for prompt-based image editing and inpainting. Simply mask an area and describe the change you want to see.
*   **OpenAI LLM:** Access OpenAI's powerful language models (GPT-4, GPT-5, o1, etc.) for text generation and reasoning.
*   **OpenAI Text-to-Speech:** Generate high-quality speech using OpenAI's TTS models.
*   **Google Imagen Generator & Edit:** Create and edit images with Google's Imagen models, with support for Vertex AI.
*   **Nano Banana:** A creative image generation node using a specialized Gemini model.
*   **Veo Video Generator:** Generate high-quality video clips from text prompts using Google's Veo model via Vertex AI or the Gemini API.
*   **ElevenLabs TTS:** Generate high-quality speech from text using ElevenLabs' diverse range of voices and models.
*   **Gemini TTS:** Create speech from text using Google's Gemini models.
*   **Tripo Text-to-3D:** Generate 3D models from text prompts using Tripo AI's advanced 3D generation API.
*   **Tripo Image-to-3D:** Convert images into detailed 3D models using Tripo AI's image-to-model capability.

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

4.  Install the required Python packages. Navigate into the newly cloned directory and use pip to install the dependencies:
    ```bash
    cd ComfyUI-ExternalAPI-Helpers
    pip install -r requirements.txt
    ```
5.  **Restart ComfyUI.** After restarting, you should find the new nodes in the "Add Node" menu.

---

## 🔑 Prerequisites: API Keys

All nodes in this collection require API keys to function.

*   **FLUX Nodes (Replicate):** You will need a [Replicate API Token](https://replicate.com/account/api-tokens).
*   **Gemini, Imagen, Nano Banana, Gemini TTS, Gemini Diarization, and Veo (Gemini API) Nodes:** You will need a [Google AI Studio API Key](https://aistudio.google.com/app/api-keys).
*   **OpenAI Nodes (GPT Image Edit, OpenAI LLM, OpenAI TTS):** You will need an [OpenAI API Key](https://platform.openai.com/api-keys).
*   **ElevenLabs TTS Node:** You will need an [ElevenLabs API Key](https://elevenlabs.io/).
*   **Tripo Nodes (Text-to-3D, Image-to-3D):** You will need a [Tripo API Key](https://tripo3d.ai/).
*   **Vertex AI Nodes (Imagen Edit, Veo Vertex AI):** You will need a Google Cloud Project ID, a service account with appropriate permissions, and the location for the resources.

You can paste your key directly into the `api_key` field on the corresponding node. For Vertex AI nodes, you will need to provide the project ID, location, and paste your service account JSON content.

### Using Environment Variables (.env file)

Instead of pasting API keys directly into nodes, you can store them in a `.env` file:

1. Copy `.env.example` to `.env` in the node directory
2. Fill in your API keys:
   ```
   GEMINI_API_KEY=your_gemini_key_here
   OPENAI_API_KEY=your_openai_key_here
   XI_API_KEY=your_elevenlabs_key_here
   REPLICATE_API_TOKEN=your_replicate_token_here
   TRIPO_API_KEY=your_tripo_api_key_here
   ```
3. In the node's `api_key` field, enter the variable name (e.g., `GEMINI_API_KEY`) instead of the actual key

This keeps your API keys secure and makes it easy to switch between different keys or projects.

---

## 📚 Node Guide

> **Note:** All nodes include standard inputs like `api_key`, `prompt`, `model`, `temperature`, and `seed` where applicable. Only unique or notable inputs are listed below.

### Flux Kontext Pro / Max

Transform images based on text prompts. Ideal for applying artistic styles or making conceptual changes.

*   **Category:** `image/edit`
*   **Key Inputs:** `image`, `aspect_ratio` (use `match_input_image` to preserve composition)
*   **Output:** `image`

### Flux.2 (Replicate)

Generate images using FLUX.2 models (Pro, Max, Dev) via Replicate.

*   **Category:** `image/generation`
*   **Key Inputs:** `image_1` to `image_5` (optional, for image-to-image tasks)
*   **Output:** `image`

### Gemini Chat

Multimodal text generation with image and audio analysis capabilities.

*   **Category:** `text/generation`
*   **Key Inputs:** `thinking`, `thinking_budget`, `system_instruction`, `image` (optional), `audio` (optional)
*   **Output:** `response` (text)

### Gemini Segmentation

Generate segmentation masks for objects in an image.

*   **Category:** `image/generation`
*   **Key Inputs:** `image`, `segment_prompt` (e.g., "the car", "all people"), `thinking`, `thinking_budget`
*   **Output:** `mask`

### Gemini Speaker Diarization

Separate audio into different speaker tracks.

*   **Category:** `audio/diarise`
*   **Key Inputs:** `audio`, `num_speakers`, `thinking`, `thinking_budget`
*   **Output:** `speaker_1` to `speaker_4` (audio tracks)

### GPT Image Edit

Prompt-based inpainting and image editing using OpenAI's API.

*   **Category:** `image/edit`
*   **Key Inputs:** `image_1` to `image_5`, `mask` (optional, white area = edit region), `background`, `quality`, `size`
*   **Output:** `image`

### OpenAI LLM

Access OpenAI language models (GPT-4, GPT-5, o1, etc.) for text generation.

*   **Category:** `text/generation`
*   **Key Inputs:** `reasoning_effort` (low/medium/high), `max_output_tokens`, `system_instruction`, `image` (optional)
*   **Output:** `response` (text)

### OpenAI Text-to-Speech

Generate speech using OpenAI's TTS models.

*   **Category:** `audio/generation`
*   **Key Inputs:** `text`, `voice` (alloy, echo, etc.), `response_format`, `speed`, `instructions` (optional)
*   **Output:** `audio`

### Google Imagen Generator

Generate images from text using Google's Imagen models.

*   **Category:** `image/generation`
*   **Key Inputs:** `number_of_images`, `aspect_ratio`, `image_size`, `guidance_scale`, `negative_prompt`
*   **Output:** `images`

### Google Imagen Edit (Vertex AI only)

Advanced image editing with inpainting, outpainting, and background swapping.

*   **Category:** `image/edit`
*   **Key Inputs:** `image`, `mask`, `project_id`, `location`, `service_account` (JSON content), `edit_mode`
*   **Output:** `edited_images`

### Nano Banana

Creative image generation using a specialized Gemini model.

*   **Category:** `image/generation`
*   **Key Inputs:** `image_1` to `image_5` (optional), `aspect_ratio`, `resolution`, `top_p`
*   **Output:** `image`

### Veo Video Generator (Vertex AI)

Generate video clips using Google's Veo model on Vertex AI.

*   **Category:** `video/generation`
*   **Key Inputs:** `project_id`, `location`, `service_account` (JSON content), `resolution`, `aspect_ratio`, `duration_seconds`, `generate_audio`, `first_frame`/`last_frame` (optional)
*   **Output:** `frames`, `audio`

### Veo Video Generator (Gemini API)

Generate videos using Veo 2.0 via the Gemini API. Supports text-to-video and image-to-video.

*   **Category:** `video/generation`
*   **Key Inputs:** `aspect_ratio`, `duration_seconds`, `negative_prompt`, `image` (optional, for image-to-video)
*   **Output:** `frames`

### ElevenLabs TTS

Generate speech using ElevenLabs' diverse voices and models.

*   **Category:** `audio/generation`
*   **Key Inputs:** `text`, `voice_id`, `model_id`, `stability`, `similarity_boost`, `speed`, `style`, `use_speaker_boost`
*   **Output:** `audio`

### Gemini TTS

Generate speech using Google's Gemini TTS models.

*   **Category:** `audio/generation`
*   **Key Inputs:** `text`, `voice_id`, `system_prompt` (optional)
*   **Output:** `audio`

### Tripo Text-to-3D

Generate high-quality 3D models from text prompts using Tripo AI's powerful generation engine.

*   **Category:** `3D/generation`
*   **Key Inputs:** `prompt`, `model_version` (v3.0, v2.5, v2.0, v1.4, Turbo), `texture_quality` (standard/detailed), `save_name`
*   **Optional Inputs:** `seed`, `face_limit` (500-100000, default 20000), `texture_size` (512/1024/2048), `pbr` (realistic lighting), `negative_prompt`
*   **Output:** `glb` (3D model file)

### Tripo Image-to-3D

Convert images into detailed 3D models using Tripo AI's advanced image-to-model technology.

*   **Category:** `3D/generation`
*   **Key Inputs:** `image`, `model_version` (v3.0, v2.5, v2.0, v1.4), `texture_quality` (standard/detailed), `save_name`
*   **Optional Inputs:** `seed`, `texture_seed` (randomness of colors/textures), `face_limit` (500-100000, default 20000), `texture_size` (512/1024/2048), `pbr` (realistic lighting)
*   **Output:** `glb` (3D model file)


##  Acknowledgements

*   The [ComfyUI](https://github.com/comfyanonymous/ComfyUI) team for creating such a flexible and powerful platform.
*   [Google](https://deepmind.google/technologies/gemini/), [OpenAI](https://openai.com/), and [Black Forest Labs](https://www.blackforestlabs.ai/) for developing these incredible models.
*   [Replicate](https://replicate.com/) for providing easy API access to a wide range of models.