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
*   **Vertex AI Nodes (Imagen Edit, Veo Vertex AI):** You will need a Google Cloud Project ID, a service account with appropriate permissions, and the location for the resources.

You can paste your key directly into the `api_key` field on the corresponding node. For Vertex AI nodes, you will need to provide the project ID, location, and path to your service account JSON file.

---

## 📚 Node Guide

### Flux Kontext Pro / Max

These nodes allow you to transform an input image based on a text prompt. They are ideal for applying artistic styles or making significant conceptual changes to an existing image.

*   **Category:** `image/edit`
*   **Inputs:**
    *   `image`: The source image to transform.
    *   `prompt`: A text description of the desired output (e.g., "A vibrant Van Gogh painting", "Make this a 90s cartoon").
    *   `replicate_api_token`: Your API token from Replicate.
    *   `aspect_ratio`: The desired output aspect ratio. `match_input_image` is highly recommended to preserve the original composition.
    *   `output_format`: `jpg` or `png`.
    *   `safety_tolerance`: Adjust the content safety filter level.
*   **Output:**
    *   `image`: The generated image.

### Flux.2 (Replicate)

Generate images using Black Forest Labs' FLUX.2 models via the Replicate API.

*   **Category:** `image/generation`
*   **Inputs:**
    *   `prompt`: The text prompt for image generation.
    *   `api_key`: Your Replicate API token.
    *   `model`: Choose between `flux-2-max`, `flux-2-pro`, or `flux-2-dev`.
    *   `aspect_ratio`: The desired aspect ratio for the generated image.
    *   `output_format`: `webp`, `jpg`, or `png`.
    *   `output_quality`: Quality of the output image (0-100).
    *   `image_1` to `image_5` (Optional): Input images for image-to-image or control tasks.
*   **Output:**
    *   `image`: The generated image.

### Gemini Chat

A versatile node for text generation and image/audio analysis. Use it to understand an image's content, analyze audio, or to generate creative text for other nodes.

*   **Category:** `text/generation`
*   **Inputs:**
    *   `prompt`: The text prompt or question you want to ask the model.
    *   `model`: The Gemini model to use (e.g., `gemini-2.5-pro`, `gemini-2.5-flash`).
    *   `temperature`: Controls the creativity of the output.
    *   `thinking`: Enables the model's thinking/reasoning process.
    *   `seed`: Seed for reproducibility.
    *   `api_key`: Your API key from Google AI Studio.
    *   `system_instruction` (Optional): Provide context or rules for how the model should behave.
    *   `thinking_budget` (Optional): Token budget for thinking.
    *   `image` (Optional): An input image for the model to analyze.
    *   `audio` (Optional): An input audio for the model to analyze.
*   **Output:**
    *   `response`: The text generated by the Gemini model.

### Gemini Segmentation

This node uses a Gemini model to generate segmentation masks for specified objects within an image.

*   **Category:** `image/generation`
*   **Inputs:**
    *   `image`: The source image for segmentation.
    *   `segment_prompt`: A text description of the objects to segment (e.g., "the car", "all people").
    *   `model`: The Gemini model to use.
    *   `temperature`: Controls randomness.
    *   `thinking`: Enable thinking process.
    *   `seed`: Seed for reproducibility.
    *   `api_key`: Your API key from Google AI Studio.
    *   `thinking_budget` (Optional): Token budget for thinking.
*   **Output:**
    *   `mask`: A black and white mask of the segmented objects.

### Gemini Speaker Diarization

Separate audio into different speaker tracks using Gemini.

*   **Category:** `audio/diarise`
*   **Inputs:**
    *   `audio`: The input audio to process.
    *   `num_speakers`: The expected number of speakers.
    *   `model`: The Gemini model to use.
    *   `api_key`: Your API key from Google AI Studio.
    *   `seed`: Seed for reproducibility.
    *   `temperature`: Controls randomness.
    *   `thinking` (Optional): Enable thinking process.
    *   `thinking_budget` (Optional): Token budget for thinking.
*   **Output:**
    *   `speaker_1` to `speaker_4`: Audio tracks for up to 4 separated speakers.

### GPT Image Edit

This node uses OpenAI's API to perform powerful, prompt-based inpainting and editing.

*   **Category:** `image/edit`
*   **Inputs:**
    *   `image`: The source image to edit.
    *   `mask` (Optional): A black and white mask. The model will edit the **white area** of the mask.
    *   `prompt`: A description of the edit to perform.
    *   `api_key`: Your API key from OpenAI.
    *   `...other_params`: Various quality and formatting options for the OpenAI API.
*   **Output:**
    *   `image`: The edited image.

### OpenAI LLM

Access OpenAI's powerful language models for text generation and reasoning.

*   **Category:** `text/generation`
*   **Inputs:**
    *   `prompt`: The text prompt.
    *   `model`: The OpenAI model to use (e.g., `gpt-4.1`, `o1`, `gpt-5`).
    *   `temperature`: Controls randomness.
    *   `reasoning_effort`: Effort level for reasoning models.
    *   `api_key`: Your OpenAI API key.
    *   `max_output_tokens`: Maximum number of tokens to generate.
    *   `system_instruction` (Optional): System level instructions.
    *   `image` (Optional): Input image for multimodal models.
*   **Output:**
    *   `response`: The generated text response.

### OpenAI Text-to-Speech

Generate high-quality speech from text using OpenAI's TTS models.

*   **Category:** `audio/generation`
*   **Inputs:**
    *   `text`: The text to convert to speech.
    *   `model`: The TTS model to use (e.g., `gpt-4o-mini-tts`, `tts-1`).
    *   `voice`: The voice to use (e.g., `alloy`, `echo`).
    *   `response_format`: Output audio format.
    *   `speed`: Speaking speed.
    *   `api_key`: Your OpenAI API key.
    *   `instructions` (Optional): Instructions for the model (supported by some models).
*   **Output:**
    *   `audio`: The generated audio.

### Google Imagen Generator

Generate images from a text prompt using Google's Imagen models.

*   **Category:** `image/generation`
*   **Inputs:**
    *   `prompt`: A text description of the image to generate.
    *   `api_key`: Your API key from Google AI Studio.
    *   `model`: The Imagen model to use.
    *   `...other_params`: Options for number of images, aspect ratio, and image size.
*   **Output:**
    *   `images`: The generated image(s).

### Google Imagen Edit (Vertex AI only)

Perform advanced image editing, inpainting, outpainting, and background swapping using Imagen on Google's Vertex AI platform.

*   **Category:** `image/edit`
*   **Inputs:**
    *   `image`: The source image to edit.
    *   `mask`: A mask defining the area to edit.
    *   `prompt`: A description of the desired edit.
    *   `project_id`: Your Google Cloud Project ID.
    *   `location`: The Google Cloud location for the model.
    *   `service_account`: Path to your Google Cloud service account JSON file.
    *   `edit_mode`: The type of edit to perform (e.g., inpainting, outpainting).
    *   `...other_params`: Controls for negative prompt, seed, and steps.
*   **Output:**
    *   `edited_images`: The edited image(s).

### Nano Banana

A creative image generation node that can take a combination of text and up to five images as input.

*   **Category:** `image/generation`
*   **Inputs:**
    *   `api_key`: Your API key from Google AI Studio.
    *   `prompt` (Optional): A text prompt.
    *   `image_1` to `image_5` (Optional): Up to five source images.
    *   `...other_params`: Controls for aspect ratio, temperature, top_p, and seed.
*   **Output:**
    *   `image`: The generated image.

### Veo Video Generator (Vertex AI)

Generate short, high-quality video clips from a text description using Google's Veo model on Vertex AI.

*   **Category:** `video/generation`
*   **Inputs:**
    *   `prompt`: A text description of the video to generate.
    *   `project_id`: Your Google Cloud Project ID.
    *   `location`: The Google Cloud location for the model.
    *   `service_account`: Path to your Google Cloud service account JSON file.
    *   `...other_params`: Controls for negative prompt, aspect ratio, audio generation, and seed.
*   **Output:**
    *   `frames`: The generated video frames, output as an image batch.

### Veo Video Generator (Gemini API)

Generate videos using Google's Veo 2.0 model via the Gemini API. Supports text-to-video and image-to-video.

*   **Category:** `video/generation`
*   **Inputs:**
    *   `prompt`: A text description of the video.
    *   `image` (Optional): An input image for image-to-video generation.
    *   `api_key`: Your API key from Google AI Studio.
    *   `model`: The Veo model to use (e.g., `veo-2.0-generate-001`).
    *   `aspect_ratio`: Desired aspect ratio (16:9 or 9:16).
    *   `duration_seconds`: Duration of the video (e.g., 5-8 seconds).
    *   `...other_params`: Controls for negative prompt and seed.
*   **Output:**
    *   `frames`: The generated video frames.

### ElevenLabs TTS

Generate speech from text using the ElevenLabs API.

*   **Category:** `audio/generation`
*   **Inputs:**
    *   `text`: The text to convert to speech.
    *   `api_key`: Your API key from ElevenLabs.
    *   `voice_id`: The ID of the voice to use for generation.
    *   `model_id`: The ElevenLabs model to use.
    *   `output_format`: The desired output audio format.
    *   `stability`: Controls the stability and variability of the generated speech.
    *   `similarity_boost`: Enhances the similarity of the generated speech to the chosen voice.
    *   `speed`: Adjusts the speaking rate.
    *   `style`: Controls the expressiveness of the speech.
    *   `use_speaker_boost`: A boolean to enable or disable speaker boost.
    *   `seed`: A seed for ensuring reproducible results.
*   **Output:**
    *   `audio`: The generated audio waveform and sample rate.

### Gemini TTS

Generate speech from text using Google's Gemini TTS models.

*   **Category:** `audio/generation`
*   **Inputs:**
    *   `text`: The text to be converted into speech.
    *   `api_key`: Your API key from Google AI Studio.
    *   `model`: The specific Gemini model to use for generation.
    *   `voice_id`: The prebuilt voice to use for the output.
    *   `temperature`: Controls the randomness and creativity of the output.
    *   `seed`: A seed for ensuring reproducible results.
    *   `system_prompt` (Optional): A system-level instruction to guide the model's behavior.
*   **Output:**
    *   `audio`: The generated audio waveform and sample rate.


##  Acknowledgements

*   The [ComfyUI](https://github.com/comfyanonymous/ComfyUI) team for creating such a flexible and powerful platform.
*   [Google](https://deepmind.google/technologies/gemini/), [OpenAI](https://openai.com/), and [Black Forest Labs](https://www.blackforestlabs.ai/) for developing these incredible models.
*   [Replicate](https://replicate.com/) for providing easy API access to a wide range of models.