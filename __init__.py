from dotenv import load_dotenv
load_dotenv()

import importlib

modules = [
    "flux_kontext_replicate",
    "gemini_node",
    "gemini_diarisation",
    "gpt_image_edit",
    "imagen",
    "imagen_edit",
    "veo",
    "veo_api",
    "gemini_segment",
    "nano_banana",
    "gemini_tts",
    "elevenlabs_tts",
    "flux2_replicate",
    "openai_tts",
    "openai_node",
    "sora",
    "cleanup",
    "tripoTextToModel",
    "tripoImageToModel",
    "grok",
    "groq_node",
    "groq_orpheus",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in modules:
    module = importlib.import_module(f".{module_name}", package=__name__)
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]