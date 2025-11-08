"""
Groq Batch Node for ComfyUI.

Provides concurrent batch processing for multiple prompts.
Uses asyncio for efficient parallel request handling.
"""

from typing import Dict, Any, List, Tuple
import json
import asyncio

from .groq_utils import (
    GroqAPIManager,
    RetryHandler,
    ResponseParser,
    ModelCache
)


class GroqBatchNode:
    """
    ComfyUI node for batch processing with Groq API.

    Processes multiple prompts concurrently with progress tracking
    and error recovery for individual requests.
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
                "prompts_json": ("STRING", {
                    "multiline": True,
                    "default": json.dumps([
                        "Tell me a joke",
                        "What is 2+2?",
                        "Name a color"
                    ], indent=2)
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
                "max_concurrent": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("responses_json", "summary", "errors_json")
    FUNCTION = "process_batch"
    CATEGORY = "groq/batch"
    OUTPUT_NODE = False

    def process_batch(
        self,
        prompts_json: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        api_key: str = "",
        system_prompt: str = "",
        max_concurrent: int = 5
    ) -> Tuple[str, str, str]:
        """
        Process multiple prompts concurrently.

        Args:
            prompts_json: JSON array of prompt strings
            model: Model ID to use
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens per response
            api_key: Optional Groq API key
            system_prompt: Optional system message for all prompts
            max_concurrent: Maximum concurrent requests

        Returns:
            Tuple of (responses_json, summary_string, errors_json)

        Raises:
            ValueError: If prompts_json is invalid
            Exception: If batch processing fails
        """
        try:
            prompts = self._parse_prompts(prompts_json)

            if not prompts:
                raise ValueError("Prompts list cannot be empty")

            responses, errors = asyncio.run(
                self._process_batch_async(
                    prompts,
                    model,
                    temperature,
                    max_tokens,
                    api_key,
                    system_prompt,
                    max_concurrent
                )
            )

            responses_json = json.dumps(responses, indent=2)
            errors_json = json.dumps(errors, indent=2)

            summary = self._generate_summary(prompts, responses, errors)

            return (responses_json, summary, errors_json)

        except ValueError as e:
            error_msg = f"Configuration error: {str(e)}"
            print(error_msg)
            return ("[]", error_msg, "[]")
        except Exception as e:
            error_msg = f"Error processing batch: {str(e)}"
            print(error_msg)
            return ("[]", error_msg, "[]")

    async def _process_batch_async(
        self,
        prompts: List[str],
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: str,
        system_prompt: str,
        max_concurrent: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process batch of prompts asynchronously with concurrency limit.

        Args:
            prompts: List of prompt strings
            model: Model ID
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            api_key: API key
            system_prompt: System message
            max_concurrent: Maximum concurrent requests

        Returns:
            Tuple of (successful_responses, errors)
        """
        client = GroqAPIManager.get_async_client(api_key if api_key else None)

        semaphore = asyncio.Semaphore(max_concurrent)

        tasks = [
            self._process_single_prompt(
                client,
                prompt,
                model,
                temperature,
                max_tokens,
                system_prompt,
                semaphore,
                index
            )
            for index, prompt in enumerate(prompts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        errors = []

        for index, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    "index": index,
                    "prompt": prompts[index],
                    "error": str(result)
                })
            elif isinstance(result, dict) and "error" in result:
                errors.append(result)
            else:
                responses.append(result)

        return responses, errors

    async def _process_single_prompt(
        self,
        client: Any,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
        semaphore: asyncio.Semaphore,
        index: int
    ) -> Dict[str, Any]:
        """
        Process a single prompt with semaphore for concurrency control.

        Args:
            client: Async Groq client
            prompt: Prompt string
            model: Model ID
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            system_prompt: System message
            semaphore: Semaphore for concurrency control
            index: Index in batch

        Returns:
            Dictionary containing prompt, response, and metadata
        """
        async with semaphore:
            try:
                messages = []

                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({"role": "user", "content": prompt})

                response = await self.retry_handler.execute_async(
                    client.chat.completions.create,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                response_text, usage_info = ResponseParser.parse_chat_completion(response)

                return {
                    "index": index,
                    "prompt": prompt,
                    "response": response_text,
                    "usage": usage_info
                }

            except Exception as e:
                return {
                    "index": index,
                    "prompt": prompt,
                    "error": str(e)
                }

    def _parse_prompts(self, prompts_json: str) -> List[str]:
        """
        Parse and validate prompts JSON.

        Args:
            prompts_json: JSON string containing prompt array

        Returns:
            List of prompt strings

        Raises:
            ValueError: If JSON is invalid or not a list
        """
        try:
            prompts = json.loads(prompts_json)

            if not isinstance(prompts, list):
                raise ValueError("Prompts must be a JSON array")

            if not all(isinstance(p, str) for p in prompts):
                raise ValueError("All prompts must be strings")

            return prompts

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompts: {e}")

    def _generate_summary(
        self,
        prompts: List[str],
        responses: List[Dict[str, Any]],
        errors: List[Dict[str, Any]]
    ) -> str:
        """
        Generate summary of batch processing results.

        Args:
            prompts: Original prompts list
            responses: Successful responses
            errors: Failed requests

        Returns:
            Summary string
        """
        total = len(prompts)
        success = len(responses)
        failed = len(errors)

        total_tokens = sum(
            resp.get("usage", {}).get("total_tokens", 0)
            for resp in responses
        )

        summary = "Batch Processing Summary:\n"
        summary += f"Total: {total}, Success: {success}, Failed: {failed}\n"
        summary += f"Total Tokens Used: {total_tokens}\n"

        if failed > 0:
            summary += f"\nFailed prompts (indices): {[e['index'] for e in errors]}"

        return summary
