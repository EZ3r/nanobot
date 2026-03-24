"""Tests for lightweight local input compression and clarification."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.input_guard import preprocess_input
from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse


def _make_loop(tmp_path: Path, *, mode: str = "guide") -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        input_guard_mode=mode,
    )


def test_preprocess_input_compresses_plaintext_but_preserves_code_blocks() -> None:
    raw = " Hello   world!!!!!\r\n\r\n\r\n```python\nx  =  1\n```\n\nNext   step??? "

    result = preprocess_input(raw, mode="light")

    assert result.clarification_message is None
    assert result.content == "Hello world!!\n\n```python\nx  =  1\n```\n\nNext step??"


def test_preprocess_input_returns_local_clarification_for_vague_request() -> None:
    result = preprocess_input("帮我看下这个", mode="guide")

    assert result.clarification_message is not None
    assert "I have not sent your last message to the model yet" in result.clarification_message
    assert "Goal | Input/Context | Expected output" in result.clarification_message


def test_preprocess_input_strict_keeps_detailed_chinese_request() -> None:
    result = preprocess_input(
        "帮我查询一下今天的天气，哦对我在桂林市灵川县三街镇，顺便也查一下明天的天气吧",
        mode="strict",
    )

    assert result.clarification_message is None


def test_preprocess_input_strict_keeps_structured_multistep_request() -> None:
    result = preprocess_input(
        "我需要你帮我做以下的事情：1:打开A文件夹；2.选中其中的a,b文件；3. 将他们剪切; 4.退出A文件夹；5. 打开B文件夹; 6.粘贴a,b文件",
        mode="strict",
    )

    assert result.clarification_message is None


def test_preprocess_input_strict_keeps_long_unpunctuated_chinese_request() -> None:
    result = preprocess_input("请帮我把桌面上的月度报表移动到归档文件夹然后重命名为三月最终版", mode="strict")

    assert result.clarification_message is None


@pytest.mark.asyncio
async def test_process_message_short_circuits_on_local_clarification(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, mode="guide")
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="should not run"))

    result = await loop._process_message(
        InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="fix this")
    )

    assert result is not None
    assert "too vague" in result.content
    loop.provider.chat_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_process_message_passes_compressed_text_into_context_builder(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, mode="light")
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="ok"))
    loop.context.build_messages = MagicMock(return_value=[])

    result = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content="  Hello   world!!!!!\n\n\nNext   step???  ",
        )
    )

    assert result is not None
    loop.context.build_messages.assert_called_once()
    assert loop.context.build_messages.call_args.kwargs["current_message"] == "Hello world!!\n\nNext step??"
