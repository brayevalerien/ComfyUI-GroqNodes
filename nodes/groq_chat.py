"""
Groq Chat Node for ComfyUI.

Provides text generation capabilities using Groq's chat completion models.
Supports conversation history, streaming, and various generation parameters.
"""

from typing import Dict, Any, List, Tuple
import json

from .groq_utils import (
    GroqAPIManager,
    RetryHandler,
    ResponseParser,
    ModelCache
)


class GroqChatNode:
    """
    ComfyUI node for Groq chat completions.

    Generates text responses using Groq's language models with support
    for conversation history, temperature control, and token limits.
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
        models = ModelCache.get_model_list("chat")

        if not models:
            models = ["llama-3.3-70b-versatile"]

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Hello! How can I help you today?"
                }),
                "model": (models, {
                    "default": models[0]
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 32768,
                    "step": 1
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "conversation_history": ("STRING", {
                    "multiline": True,
                    "default": "[]"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "usage_info", "conversation_history")
    FUNCTION = "generate"
    CATEGORY = "groq/language"
    OUTPUT_NODE = False

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        api_key: str = "",
        system_prompt: str = "",
        conversation_history: str = "[]",
        seed: int = -1
    ) -> Tuple[str, str, str]:
        """
        Generate text completion using Groq API.

        Args:
            prompt: User input text
            model: Model ID to use for generation
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            api_key: Optional Groq API key
            system_prompt: Optional system message
            conversation_history: JSON string of previous messages
            seed: Random seed for reproducibility (-1 for random)

        Returns:
            Tuple of (response_text, usage_info_string, updated_conversation_history)

        Raises:
            ValueError: If API key is invalid or parameters are incorrect
            Exception: If API request fails after retries
        """
        try:
            client = GroqAPIManager.get_client(api_key if api_key else None)

            messages = self._build_messages(
                prompt,
                system_prompt,
                conversation_history
            )

            request_params = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            if seed >= 0:
                request_params["seed"] = seed

            response = self.retry_handler.execute(
                client.chat.completions.create,
                **request_params
            )

            response_text, usage_info = ResponseParser.parse_chat_completion(response)

            updated_history = self._update_conversation_history(
                messages,
                response_text
            )

            usage_string = ResponseParser.format_usage_info(usage_info)

            return (response_text, usage_string, json.dumps(updated_history, indent=2))

        except ValueError as e:
            error_msg = f"Configuration error: {str(e)}"
            print(error_msg)
            return (error_msg, "", "[]")
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return (error_msg, "", conversation_history)

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str,
        conversation_history: str
    ) -> List[Dict[str, str]]:
        """
        Build messages array from inputs.

        Args:
            prompt: Current user prompt
            system_prompt: Optional system message
            conversation_history: JSON string of previous messages

        Returns:
            List of message dictionaries
        """
        messages = []

        try:
            history = json.loads(conversation_history)
            if isinstance(history, list):
                messages.extend(history)
        except (json.JSONDecodeError, TypeError):
            pass

        if system_prompt and not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return messages

    def _update_conversation_history(
        self,
        messages: List[Dict[str, str]],
        response: str
    ) -> List[Dict[str, str]]:
        """
        Update conversation history with assistant response.

        Args:
            messages: Current message list
            response: Assistant's response

        Returns:
            Updated message list
        """
        updated = messages.copy()
        updated.append({"role": "assistant", "content": response})
        return updated
