from .nodes.groq_chat import GroqChatNode
from .nodes.groq_vision import GroqVisionNode
from .nodes.groq_tool_use import GroqToolUseNode
from .nodes.groq_audio import GroqAudioNode
from .nodes.groq_batch import GroqBatchNode

NODE_CLASS_MAPPINGS = {
    "GroqChat": GroqChatNode,
    "GroqVision": GroqVisionNode,
    "GroqToolUse": GroqToolUseNode,
    "GroqAudio": GroqAudioNode,
    "GroqBatch": GroqBatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqChat": "Groq Chat",
    "GroqVision": "Groq Vision",
    "GroqToolUse": "Groq Tool Use",
    "GroqAudio": "Groq Audio",
    "GroqBatch": "Groq Batch"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
