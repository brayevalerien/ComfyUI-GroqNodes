import pytest
import json
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nodes.groq_chat import GroqChatNode
from nodes.groq_vision import GroqVisionNode
from nodes.groq_tool_use import GroqToolUseNode


class TestGroqChatNode:

    @pytest.fixture
    def mock_groq_client(self):
        with patch("nodes.groq_chat.GroqAPIManager.get_client") as mock:
            client = Mock()
            mock.return_value = client

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15
            mock_response.model = "test-model"

            client.chat.completions.create.return_value = mock_response
            yield client

    def test_input_types(self):
        input_types = GroqChatNode.INPUT_TYPES()

        assert "required" in input_types
        assert "prompt" in input_types["required"]
        assert "model" in input_types["required"]
        assert "optional" in input_types

    def test_generate_basic(self, mock_groq_client):
        node = GroqChatNode()

        text, usage, history = node.generate(
            prompt="Hello",
            model="test-model",
            temperature=1.0,
            max_tokens=1024,
            top_p=1.0
        )

        assert text == "Test response"
        assert "test-model" in usage
        assert isinstance(history, str)

    def test_generate_with_system_prompt(self, mock_groq_client):
        node = GroqChatNode()

        text, usage, history = node.generate(
            prompt="Hello",
            model="test-model",
            system_prompt="You are a helpful assistant",
            temperature=1.0,
            max_tokens=1024,
            top_p=1.0
        )

        call_args = mock_groq_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"

    def test_generate_with_conversation_history(self, mock_groq_client):
        node = GroqChatNode()
        history = json.dumps([
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"}
        ])

        text, usage, new_history = node.generate(
            prompt="New message",
            model="test-model",
            conversation_history=history,
            temperature=1.0,
            max_tokens=1024,
            top_p=1.0
        )

        call_args = mock_groq_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) >= 3
        assert messages[-1]["content"] == "New message"

    def test_generate_with_seed(self, mock_groq_client):
        node = GroqChatNode()

        node.generate(
            prompt="Hello",
            model="test-model",
            seed=42,
            temperature=1.0,
            max_tokens=1024,
            top_p=1.0
        )

        call_args = mock_groq_client.chat.completions.create.call_args
        assert call_args.kwargs["seed"] == 42

    def test_generate_error_handling(self):
        with patch("nodes.groq_chat.GroqAPIManager.get_client") as mock:
            mock.side_effect = ValueError("Invalid API key")

            node = GroqChatNode()
            text, usage, history = node.generate(
                prompt="Hello",
                model="test-model",
                temperature=1.0,
                max_tokens=1024,
                top_p=1.0
            )

            assert "Configuration error" in text
            assert usage == ""


class TestGroqVisionNode:

    @pytest.fixture
    def mock_groq_client(self):
        with patch("nodes.groq_vision.GroqAPIManager.get_client") as mock:
            client = Mock()
            mock.return_value = client

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "I see an image"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            mock_response.model = "vision-model"

            client.chat.completions.create.return_value = mock_response
            yield client

    @pytest.fixture
    def sample_image_tensor(self):
        image_np = np.random.rand(1, 256, 256, 3).astype(np.float32)
        return torch.from_numpy(image_np)

    def test_input_types(self):
        input_types = GroqVisionNode.INPUT_TYPES()

        assert "required" in input_types
        assert "image" in input_types["required"]
        assert "prompt" in input_types["required"]
        assert "model" in input_types["required"]

    def test_analyze_basic(self, mock_groq_client, sample_image_tensor):
        node = GroqVisionNode()

        text, usage = node.analyze(
            image=sample_image_tensor,
            prompt="What is in this image?",
            model="vision-model",
            temperature=1.0,
            max_tokens=1024
        )

        assert text == "I see an image"
        assert "vision-model" in usage

    def test_analyze_builds_correct_messages(self, mock_groq_client, sample_image_tensor):
        node = GroqVisionNode()

        node.analyze(
            image=sample_image_tensor,
            prompt="Describe this",
            model="vision-model",
            temperature=1.0,
            max_tokens=1024
        )

        call_args = mock_groq_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) > 0
        content = messages[-1]["content"]
        assert any(item["type"] == "text" for item in content)
        assert any(item["type"] == "image_url" for item in content)

    def test_analyze_batch_images(self, mock_groq_client):
        node = GroqVisionNode()
        batch_images = torch.from_numpy(
            np.random.rand(3, 128, 128, 3).astype(np.float32)
        )

        text, usage = node.analyze(
            image=batch_images,
            prompt="Describe these images",
            model="vision-model",
            temperature=1.0,
            max_tokens=1024
        )

        call_args = mock_groq_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        content = messages[-1]["content"]

        image_urls = [item for item in content if item["type"] == "image_url"]
        assert len(image_urls) == 3


class TestGroqToolUseNode:

    @pytest.fixture
    def mock_groq_client(self):
        with patch("nodes.groq_tool_use.GroqAPIManager.get_client") as mock:
            client = Mock()
            mock.return_value = client

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = None

            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "get_weather"
            mock_tool_call.function.arguments = '{"location": "SF", "unit": "celsius"}'
            mock_response.choices[0].message.tool_calls = [mock_tool_call]

            mock_response.choices[0].finish_reason = "tool_calls"
            mock_response.usage.prompt_tokens = 20
            mock_response.usage.completion_tokens = 10
            mock_response.usage.total_tokens = 30
            mock_response.model = "test-model"

            client.chat.completions.create.return_value = mock_response
            yield client

    @pytest.fixture
    def sample_tools_json(self):
        return json.dumps([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }])

    def test_input_types(self):
        input_types = GroqToolUseNode.INPUT_TYPES()

        assert "required" in input_types
        assert "prompt" in input_types["required"]
        assert "tools_json" in input_types["required"]

    def test_execute_with_tools_basic(self, mock_groq_client, sample_tools_json):
        node = GroqToolUseNode()

        text, tool_calls, usage = node.execute_with_tools(
            prompt="What's the weather?",
            tools_json=sample_tools_json,
            model="test-model",
            temperature=1.0
        )

        assert "tool call" in text.lower()
        tool_calls_data = json.loads(tool_calls)
        assert len(tool_calls_data) == 1
        assert tool_calls_data[0]["name"] == "get_weather"

    def test_execute_with_invalid_tools_json(self):
        node = GroqToolUseNode()

        text, tool_calls, usage = node.execute_with_tools(
            prompt="Test",
            tools_json="invalid json",
            model="test-model",
            temperature=1.0
        )

        assert "Configuration error" in text

    def test_parse_tools_validates_structure(self):
        node = GroqToolUseNode()

        with pytest.raises(ValueError, match="must be a JSON array"):
            node._parse_tools('{"type": "function"}')

        with pytest.raises(ValueError, match="type must be 'function'"):
            node._parse_tools('[{"type": "invalid"}]')
