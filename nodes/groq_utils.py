"""
Base utilities for Groq API integration with ComfyUI.

This module provides core functionality for API management, error handling,
and data conversion utilities used across all Groq nodes.
"""

import os
import json
import time
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable

import numpy as np
from PIL import Image
from groq import Groq, AsyncGroq
from groq.types.chat import ChatCompletion
from dotenv import load_dotenv

load_dotenv()


class GroqAPIManager:
    """
    Manages Groq API client instances and handles API key validation.

    Supports API key retrieval from environment variables or direct input.
    Implements singleton pattern for efficient client reuse.
    """

    _client_cache: Dict[str, Groq] = {}
    _async_client_cache: Dict[str, AsyncGroq] = {}

    @classmethod
    def get_api_key(cls, api_key_input: Optional[str] = None) -> str:
        """
        Retrieve API key from input or environment variable.

        Args:
            api_key_input: Optional API key provided directly

        Returns:
            Valid API key string

        Raises:
            ValueError: If no API key is found
        """
        api_key = api_key_input or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or provide it in the node interface."
            )
        return api_key.strip()

    @classmethod
    def get_client(cls, api_key: Optional[str] = None) -> Groq:
        """
        Get or create a synchronous Groq client instance.

        Args:
            api_key: Optional API key, will use environment variable if not provided

        Returns:
            Groq client instance
        """
        api_key = cls.get_api_key(api_key)

        if api_key not in cls._client_cache:
            cls._client_cache[api_key] = Groq(api_key=api_key)

        return cls._client_cache[api_key]

    @classmethod
    def get_async_client(cls, api_key: Optional[str] = None) -> AsyncGroq:
        """
        Get or create an asynchronous Groq client instance.

        Args:
            api_key: Optional API key, will use environment variable if not provided

        Returns:
            AsyncGroq client instance
        """
        api_key = cls.get_api_key(api_key)

        if api_key not in cls._async_client_cache:
            cls._async_client_cache[api_key] = AsyncGroq(api_key=api_key)

        return cls._async_client_cache[api_key]

    @classmethod
    def clear_cache(cls):
        """Clear all cached client instances."""
        cls._client_cache.clear()
        cls._async_client_cache.clear()


class RetryHandler:
    """
    Implements exponential backoff retry logic for API calls.

    Handles rate limits, temporary failures, and network issues
    with configurable retry parameters.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff calculation
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given retry attempt using exponential backoff.

        Args:
            attempt: Current retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if request should be retried based on exception type.

        Args:
            exception: The exception that occurred
            attempt: Current attempt number

        Returns:
            True if should retry, False otherwise
        """
        if attempt >= self.max_retries:
            return False

        error_msg = str(exception).lower()

        retryable_errors = [
            "rate limit",
            "timeout",
            "connection",
            "503",
            "502",
            "500",
            "429"
        ]

        return any(err in error_msg for err in retryable_errors)

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if not self.should_retry(e, attempt):
                    raise

                delay = self.get_delay(attempt)
                print(f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)

        raise last_exception

    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute async function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if not self.should_retry(e, attempt):
                    raise

                delay = self.get_delay(attempt)
                print(f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                print(f"Retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)

        raise last_exception


class ResponseParser:
    """
    Utility class for parsing and extracting data from Groq API responses.
    """

    @staticmethod
    def parse_chat_completion(response: ChatCompletion) -> Tuple[str, Dict[str, Any]]:
        """
        Parse chat completion response.

        Args:
            response: ChatCompletion object from Groq API

        Returns:
            Tuple of (response_text, usage_info)
        """
        text = response.choices[0].message.content or ""

        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason
        }

        return text, usage_info

    @staticmethod
    def parse_tool_calls(response: ChatCompletion) -> List[Dict[str, Any]]:
        """
        Extract tool calls from response.

        Args:
            response: ChatCompletion object from Groq API

        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        message = response.choices[0].message

        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                })

        return tool_calls

    @staticmethod
    def format_usage_info(usage_info: Dict[str, Any]) -> str:
        """
        Format usage information for display.

        Args:
            usage_info: Dictionary containing usage statistics

        Returns:
            Formatted string
        """
        return (
            f"Model: {usage_info.get('model', 'unknown')}\n"
            f"Tokens - Prompt: {usage_info.get('prompt_tokens', 0)}, "
            f"Completion: {usage_info.get('completion_tokens', 0)}, "
            f"Total: {usage_info.get('total_tokens', 0)}\n"
            f"Finish Reason: {usage_info.get('finish_reason', 'unknown')}"
        )


class ImageConverter:
    """
    Handles conversion between ComfyUI image tensors and base64 encoded images.
    """

    @staticmethod
    def tensor_to_pil(tensor: np.ndarray) -> Image.Image:
        """
        Convert ComfyUI image tensor to PIL Image.

        Args:
            tensor: NumPy array with shape (H, W, C) and values in [0, 1]

        Returns:
            PIL Image object
        """
        if tensor.ndim == 4:
            tensor = tensor[0]

        image_np = (tensor * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    @staticmethod
    def pil_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 95) -> str:
        """
        Convert PIL Image to base64 encoded string.

        Args:
            image: PIL Image object
            format: Image format (JPEG, PNG, etc.)
            quality: JPEG quality (1-100)

        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()

        if format.upper() == "JPEG" and image.mode == "RGBA":
            image = image.convert("RGB")

        image.save(buffered, format=format, quality=quality)
        img_bytes = buffered.getvalue()

        return base64.b64encode(img_bytes).decode("utf-8")

    @staticmethod
    def tensor_to_base64(
        tensor: np.ndarray,
        format: str = "JPEG",
        quality: int = 95
    ) -> str:
        """
        Convert ComfyUI image tensor directly to base64.

        Args:
            tensor: NumPy array with shape (H, W, C) or (B, H, W, C)
            format: Image format (JPEG, PNG, etc.)
            quality: JPEG quality (1-100)

        Returns:
            Base64 encoded string
        """
        pil_image = ImageConverter.tensor_to_pil(tensor)
        return ImageConverter.pil_to_base64(pil_image, format, quality)

    @staticmethod
    def batch_tensors_to_base64(
        tensor: np.ndarray,
        format: str = "JPEG",
        quality: int = 95
    ) -> List[str]:
        """
        Convert batch of image tensors to list of base64 strings.

        Args:
            tensor: NumPy array with shape (B, H, W, C)
            format: Image format (JPEG, PNG, etc.)
            quality: JPEG quality (1-100)

        Returns:
            List of base64 encoded strings
        """
        if tensor.ndim != 4:
            return [ImageConverter.tensor_to_base64(tensor, format, quality)]

        base64_images = []
        for i in range(tensor.shape[0]):
            base64_img = ImageConverter.tensor_to_base64(tensor[i], format, quality)
            base64_images.append(base64_img)

        return base64_images


class ModelCache:
    """
    Caches model information to avoid repeated file reads.
    """

    _cache: Optional[Dict[str, Any]] = None
    _cache_path: Optional[Path] = None

    @classmethod
    def load_models(cls, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load model configuration from JSON file.

        Args:
            force_reload: Force reload from file even if cached

        Returns:
            Dictionary containing model configurations
        """
        if cls._cache is not None and not force_reload:
            return cls._cache

        if cls._cache_path is None:
            current_dir = Path(__file__).parent.parent
            cls._cache_path = current_dir / "configs" / "models.json"

        try:
            with open(cls._cache_path, 'r') as f:
                cls._cache = json.load(f)
            return cls._cache
        except Exception as e:
            print(f"Error loading model config: {e}")
            return {
                "chat_models": [],
                "vision_models": [],
                "audio_models": []
            }

    @classmethod
    def get_model_list(cls, model_type: str) -> List[str]:
        """
        Get list of model IDs for specified type.

        Args:
            model_type: Type of models to retrieve (chat, vision, audio)

        Returns:
            List of model ID strings
        """
        models = cls.load_models()
        model_key = f"{model_type}_models"

        if model_key not in models:
            return []

        return [model["id"] for model in models[model_key]]

    @classmethod
    def get_model_info(cls, model_id: str, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: ID of the model
            model_type: Type of model (chat, vision, audio)

        Returns:
            Model information dictionary or None if not found
        """
        models = cls.load_models()
        model_key = f"{model_type}_models"

        if model_key not in models:
            return None

        for model in models[model_key]:
            if model["id"] == model_id:
                return model

        return None


def validate_json_schema(schema: str) -> Dict[str, Any]:
    """
    Validate and parse JSON schema string.

    Args:
        schema: JSON schema as string

    Returns:
        Parsed schema dictionary

    Raises:
        ValueError: If schema is invalid
    """
    try:
        parsed = json.loads(schema)

        if not isinstance(parsed, dict):
            raise ValueError("Schema must be a JSON object")

        if "type" not in parsed:
            raise ValueError("Schema must have a 'type' field")

        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON schema: {e}")
