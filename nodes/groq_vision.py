"""
Groq Vision Node for ComfyUI.

Provides image analysis capabilities using Groq's vision-enabled models.
Supports batch image processing and multi-image conversations.
"""

from typing import Any, Dict, List, Tuple

import torch

from .groq_utils import GroqAPIManager, ImageConverter, ModelCache, ResponseParser, RetryHandler


class GroqVisionNode:
    """
    ComfyUI node for Groq vision completions.

    Analyzes images using Groq's vision-capable models, supporting
    both single and batch image processing with customizable prompts.
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
        models = ModelCache.get_model_list("vision")

        if not models:
            models = ["llama-3.2-11b-vision-preview"]

        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "What is in this image?"}),
                "model": (models, {"default": models[0]}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1, "display": "slider"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "jpeg_quality": ("INT", {"default": 95, "min": 50, "max": 100, "step": 5}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "usage_info")
    FUNCTION = "analyze"
    CATEGORY = "groq/vision"
    OUTPUT_NODE = False

    def analyze(
        self,
        image: torch.Tensor,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        api_key: str = "",
        system_prompt: str = "",
        jpeg_quality: int = 95,
    ) -> Tuple[str, str]:
        """
        Analyze image(s) using Groq vision API.

        Args:
            image: ComfyUI IMAGE tensor (B, H, W, C) with values in [0, 1]
            prompt: Question or instruction about the image
            model: Vision model ID to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            api_key: Optional Groq API key
            system_prompt: Optional system message
            jpeg_quality: JPEG compression quality (50-100)

        Returns:
            Tuple of (response_text, usage_info_string)

        Raises:
            ValueError: If API key is invalid or parameters are incorrect
            Exception: If API request fails after retries
        """
        try:
            client = GroqAPIManager.get_client(api_key if api_key else None)

            image_np = image.cpu().numpy()

            base64_images = ImageConverter.batch_tensors_to_base64(image_np, format="JPEG", quality=jpeg_quality)

            messages = self._build_vision_messages(prompt, base64_images, system_prompt)

            request_params = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            response = self.retry_handler.execute(client.chat.completions.create, **request_params)

            response_text, usage_info = ResponseParser.parse_chat_completion(response)
            usage_string = ResponseParser.format_usage_info(usage_info)

            return (response_text, usage_string)

        except ValueError as e:
            error_msg = f"Configuration error: {str(e)}"
            print(error_msg)
            return (error_msg, "")
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            print(error_msg)
            return (error_msg, "")

    def _build_vision_messages(self, prompt: str, base64_images: List[str], system_prompt: str) -> List[Dict[str, Any]]:
        """
        Build messages array for vision API.

        Args:
            prompt: User prompt text
            base64_images: List of base64 encoded images
            system_prompt: Optional system message

        Returns:
            List of message dictionaries with content arrays
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = [{"type": "text", "text": prompt}]

        for base64_image in base64_images:
            image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}

            content.append(image_content)

        messages.append({"role": "user", "content": content})

        return messages
