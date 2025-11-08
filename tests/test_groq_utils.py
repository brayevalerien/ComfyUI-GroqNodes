import pytest
import os
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes.groq_utils import (
    GroqAPIManager,
    RetryHandler,
    ResponseParser,
    ImageConverter,
    ModelCache,
    validate_json_schema
)


class TestGroqAPIManager:

    def test_get_api_key_from_input(self):
        api_key = GroqAPIManager.get_api_key("test_key_123")
        assert api_key == "test_key_123"

    def test_get_api_key_from_env(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "env_key_456"}):
            api_key = GroqAPIManager.get_api_key()
            assert api_key == "env_key_456"

    def test_get_api_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                GroqAPIManager.get_api_key()

    def test_get_api_key_strips_whitespace(self):
        api_key = GroqAPIManager.get_api_key("  test_key  ")
        assert api_key == "test_key"


class TestRetryHandler:

    def test_execute_success_first_try(self):
        handler = RetryHandler(max_retries=3)
        mock_func = Mock(return_value="success")

        result = handler.execute(mock_func, arg1="test")

        assert result == "success"
        assert mock_func.call_count == 1

    def test_execute_retry_on_rate_limit(self):
        handler = RetryHandler(max_retries=2, base_delay=0.01)
        mock_func = Mock(side_effect=[
            Exception("rate limit exceeded"),
            "success"
        ])

        result = handler.execute(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    def test_execute_fail_after_max_retries(self):
        handler = RetryHandler(max_retries=2, base_delay=0.01)
        mock_func = Mock(side_effect=Exception("rate limit exceeded"))

        with pytest.raises(Exception, match="rate limit"):
            handler.execute(mock_func)

        assert mock_func.call_count == 3

    def test_should_retry_non_retryable_error(self):
        handler = RetryHandler()

        should_retry = handler.should_retry(
            Exception("invalid parameter"),
            attempt=0
        )

        assert should_retry is False

    def test_get_delay_exponential(self):
        handler = RetryHandler(base_delay=1.0, exponential_base=2.0)

        assert handler.get_delay(0) == 1.0
        assert handler.get_delay(1) == 2.0
        assert handler.get_delay(2) == 4.0


class TestResponseParser:

    def test_parse_chat_completion(self):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "test-model"

        text, usage_info = ResponseParser.parse_chat_completion(mock_response)

        assert text == "Hello, world!"
        assert usage_info["prompt_tokens"] == 10
        assert usage_info["completion_tokens"] == 5
        assert usage_info["total_tokens"] == 15
        assert usage_info["model"] == "test-model"

    def test_parse_tool_calls(self):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "SF"}'
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        tool_calls = ResponseParser.parse_tool_calls(mock_response)

        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_123"
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["arguments"]["location"] == "SF"

    def test_format_usage_info(self):
        usage_info = {
            "model": "test-model",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "finish_reason": "stop"
        }

        formatted = ResponseParser.format_usage_info(usage_info)

        assert "test-model" in formatted
        assert "10" in formatted
        assert "5" in formatted
        assert "15" in formatted


class TestImageConverter:

    def test_tensor_to_pil(self):
        tensor = np.random.rand(256, 256, 3).astype(np.float32)

        pil_image = ImageConverter.tensor_to_pil(tensor)

        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (256, 256)

    def test_tensor_to_pil_batch(self):
        tensor = np.random.rand(2, 256, 256, 3).astype(np.float32)

        pil_image = ImageConverter.tensor_to_pil(tensor)

        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (256, 256)

    def test_pil_to_base64(self):
        pil_image = Image.new("RGB", (100, 100), color="red")

        base64_str = ImageConverter.pil_to_base64(pil_image)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

    def test_tensor_to_base64(self):
        tensor = np.random.rand(256, 256, 3).astype(np.float32)

        base64_str = ImageConverter.tensor_to_base64(tensor)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

    def test_batch_tensors_to_base64(self):
        tensor = np.random.rand(3, 128, 128, 3).astype(np.float32)

        base64_list = ImageConverter.batch_tensors_to_base64(tensor)

        assert len(base64_list) == 3
        assert all(isinstance(s, str) for s in base64_list)


class TestModelCache:

    def test_load_models(self):
        models = ModelCache.load_models()

        assert "chat_models" in models
        assert "vision_models" in models
        assert "audio_models" in models

    def test_get_model_list(self):
        chat_models = ModelCache.get_model_list("chat")

        assert isinstance(chat_models, list)
        assert len(chat_models) > 0

    def test_get_model_info(self):
        models = ModelCache.load_models()
        first_model_id = models["chat_models"][0]["id"]

        model_info = ModelCache.get_model_info(first_model_id, "chat")

        assert model_info is not None
        assert model_info["id"] == first_model_id


class TestValidateJsonSchema:

    def test_valid_schema(self):
        schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'

        result = validate_json_schema(schema)

        assert result["type"] == "object"

    def test_invalid_json(self):
        schema = '{"type": "object"'

        with pytest.raises(ValueError, match="Invalid JSON"):
            validate_json_schema(schema)

    def test_missing_type_field(self):
        schema = '{"properties": {"name": {"type": "string"}}}'

        with pytest.raises(ValueError, match="must have a 'type'"):
            validate_json_schema(schema)
