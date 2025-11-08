"""
Groq Audio Node for ComfyUI.

Provides audio transcription capabilities using Groq's Whisper models.
Supports various audio formats and returns timing metadata.
"""

from typing import Dict, Any, Tuple
from pathlib import Path

from .groq_utils import (
    GroqAPIManager,
    RetryHandler,
    ModelCache
)


class GroqAudioNode:
    """
    ComfyUI node for Groq audio transcription.

    Transcribes audio files using Groq's Whisper models with support
    for multiple audio formats and optional timing information.
    """

    def __init__(self):
        self.retry_handler = RetryHandler(max_retries=3)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define input parameters for the node.

        Returns:
            Dictionary containing required and optional input specifications
        """
        models = ModelCache.get_model_list("audio")

        if not models:
            models = ["whisper-large-v3"]

        return {
            "required": {
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "model": (models, {
                    "default": models[0]
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "language": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "response_format": (["json", "text", "verbose_json"], {
                    "default": "json"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("transcription", "metadata")
    FUNCTION = "transcribe"
    CATEGORY = "groq/audio"
    OUTPUT_NODE = False

    def transcribe(
        self,
        audio_path: str,
        model: str,
        api_key: str = "",
        language: str = "",
        prompt: str = "",
        response_format: str = "json",
        temperature: float = 0.0
    ) -> Tuple[str, str]:
        """
        Transcribe audio file using Groq Whisper API.

        Args:
            audio_path: Path to audio file
            model: Whisper model ID to use
            api_key: Optional Groq API key
            language: Optional language code (e.g., 'en', 'es')
            prompt: Optional prompt to guide transcription
            response_format: Response format (json/text/verbose_json)
            temperature: Sampling temperature (0-1)

        Returns:
            Tuple of (transcription_text, metadata_json)

        Raises:
            ValueError: If audio path is invalid or API key missing
            Exception: If API request fails after retries
        """
        try:
            if not audio_path:
                raise ValueError("Audio path is required")

            audio_file_path = Path(audio_path)

            if not audio_file_path.exists():
                raise ValueError(f"Audio file not found: {audio_path}")

            if not audio_file_path.is_file():
                raise ValueError(f"Path is not a file: {audio_path}")

            client = GroqAPIManager.get_client(api_key if api_key else None)

            request_params = {
                "file": audio_file_path,
                "model": model,
                "response_format": response_format,
                "temperature": temperature
            }

            if language:
                request_params["language"] = language

            if prompt:
                request_params["prompt"] = prompt

            response = self.retry_handler.execute(
                client.audio.transcriptions.create,
                **request_params
            )

            transcription_text = self._extract_text(response, response_format)
            metadata = self._extract_metadata(response, response_format)

            return (transcription_text, metadata)

        except ValueError as e:
            error_msg = f"Configuration error: {str(e)}"
            print(error_msg)
            return (error_msg, "")
        except Exception as e:
            error_msg = f"Error transcribing audio: {str(e)}"
            print(error_msg)
            return (error_msg, "")

    def _extract_text(self, response: Any, response_format: str) -> str:
        """
        Extract transcription text from response.

        Args:
            response: API response object
            response_format: Format of the response

        Returns:
            Transcription text string
        """
        if response_format == "text":
            return response

        if hasattr(response, 'text'):
            return response.text

        if isinstance(response, dict) and 'text' in response:
            return response['text']

        return str(response)

    def _extract_metadata(self, response: Any, response_format: str) -> str:
        """
        Extract metadata from response.

        Args:
            response: API response object
            response_format: Format of the response

        Returns:
            JSON string containing metadata
        """
        import json

        if response_format == "text":
            return json.dumps({"format": "text"})

        metadata = {}

        if hasattr(response, 'duration'):
            metadata['duration'] = response.duration

        if hasattr(response, 'language'):
            metadata['language'] = response.language

        if response_format == "verbose_json":
            if hasattr(response, 'segments'):
                metadata['segments'] = [
                    {
                        'start': seg.start if hasattr(seg, 'start') else None,
                        'end': seg.end if hasattr(seg, 'end') else None,
                        'text': seg.text if hasattr(seg, 'text') else None
                    }
                    for seg in response.segments
                ]

            if hasattr(response, 'words'):
                metadata['words'] = [
                    {
                        'start': word.start if hasattr(word, 'start') else None,
                        'end': word.end if hasattr(word, 'end') else None,
                        'word': word.word if hasattr(word, 'word') else None
                    }
                    for word in response.words
                ]

        return json.dumps(metadata, indent=2)


NODE_CLASS_MAPPINGS = {
    "GroqAudio": GroqAudioNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqAudio": "Groq Audio"
}
