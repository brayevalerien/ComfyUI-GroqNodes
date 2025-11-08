"""
Groq Tool Use Node for ComfyUI.

Provides function calling capabilities with JSON schema validation.
Supports parallel tool execution and structured outputs.
"""

from typing import Dict, Any, List, Tuple
import json

from .groq_utils import (
    GroqAPIManager,
    RetryHandler,
    ResponseParser,
    ModelCache,
    validate_json_schema
)


class GroqToolUseNode:
    """
    ComfyUI node for Groq tool/function calling.

    Enables structured function calling with JSON schema definitions,
    allowing models to invoke tools and return structured outputs.
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
                    "default": "What is the weather in San Francisco?"
                }),
                "tools_json": ("STRING", {
                    "multiline": True,
                    "default": json.dumps([{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the current weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City name"
                                    },
                                    "unit": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"]
                                    }
                                },
                                "required": ["location"]
                            }
                        }
                    }], indent=2)
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
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "tool_choice": (["auto", "required", "none"], {
                    "default": "auto"
                }),
                "parallel_tool_calls": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response_text", "tool_calls_json", "usage_info")
    FUNCTION = "execute_with_tools"
    CATEGORY = "groq/tools"
    OUTPUT_NODE = False

    def execute_with_tools(
        self,
        prompt: str,
        tools_json: str,
        model: str,
        temperature: float = 1.0,
        api_key: str = "",
        tool_choice: str = "auto",
        parallel_tool_calls: bool = True
    ) -> Tuple[str, str, str]:
        """
        Execute chat completion with tool calling.

        Args:
            prompt: User prompt that may trigger tool calls
            tools_json: JSON array of tool definitions
            model: Model ID to use
            temperature: Sampling temperature (0-2)
            api_key: Optional Groq API key
            tool_choice: How to use tools (auto/required/none)
            parallel_tool_calls: Allow parallel tool execution

        Returns:
            Tuple of (response_text, tool_calls_json, usage_info_string)

        Raises:
            ValueError: If tool definitions are invalid
            Exception: If API request fails after retries
        """
        try:
            client = GroqAPIManager.get_client(api_key if api_key else None)

            tools = self._parse_tools(tools_json)

            messages = [{"role": "user", "content": prompt}]

            request_params = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "tools": tools,
                "tool_choice": tool_choice,
                "parallel_tool_calls": parallel_tool_calls
            }

            response = self.retry_handler.execute(
                client.chat.completions.create,
                **request_params
            )

            tool_calls = ResponseParser.parse_tool_calls(response)

            response_text = response.choices[0].message.content or ""

            if not response_text and tool_calls:
                response_text = f"Model requested {len(tool_calls)} tool call(s)"

            tool_calls_json = json.dumps(tool_calls, indent=2)

            _, usage_info = ResponseParser.parse_chat_completion(response)
            usage_string = ResponseParser.format_usage_info(usage_info)

            return (response_text, tool_calls_json, usage_string)

        except ValueError as e:
            error_msg = f"Configuration error: {str(e)}"
            print(error_msg)
            return (error_msg, "[]", "")
        except Exception as e:
            error_msg = f"Error executing with tools: {str(e)}"
            print(error_msg)
            return (error_msg, "[]", "")

    def _parse_tools(self, tools_json: str) -> List[Dict[str, Any]]:
        """
        Parse and validate tool definitions.

        Args:
            tools_json: JSON string containing tool definitions

        Returns:
            List of validated tool dictionaries

        Raises:
            ValueError: If tool definitions are invalid
        """
        try:
            tools = json.loads(tools_json)

            if not isinstance(tools, list):
                raise ValueError("Tools must be a JSON array")

            for tool in tools:
                if not isinstance(tool, dict):
                    raise ValueError("Each tool must be an object")

                if tool.get("type") != "function":
                    raise ValueError("Tool type must be 'function'")

                if "function" not in tool:
                    raise ValueError("Tool must have 'function' field")

                func = tool["function"]

                if "name" not in func:
                    raise ValueError("Function must have 'name' field")

                if "parameters" in func:
                    validate_json_schema(json.dumps(func["parameters"]))

            return tools

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tools definition: {e}")


NODE_CLASS_MAPPINGS = {
    "GroqToolUse": GroqToolUseNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqToolUse": "Groq Tool Use"
}
