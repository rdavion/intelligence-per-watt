"""Unit tests for Terminus telemetry instrumentation."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from ipw.core.registry import AgentRegistry
from ipw.telemetry.events import EventRecorder, EventType


def test_terminus_records_internal_lm_and_terminal_events() -> None:
    fake_llm = MagicMock()
    fake_llm.count_tokens.side_effect = (
        lambda value: max(1, len(str(value)) // 4) if value is not None else 0
    )
    fake_llm.call.return_value = "answer"
    mock_agent = MagicMock()
    mock_agent._llm = fake_llm

    def perform_task(input_text: str, *, session, **kwargs):
        session.send_keys("echo hello")
        return mock_agent._llm.call(prompt=input_text)

    mock_agent.perform_task.side_effect = perform_task
    mock_terminus_cls = MagicMock(return_value=mock_agent)

    with patch.dict(
        "sys.modules",
        {
            "docker": MagicMock(),
            "terminal_bench": MagicMock(),
            "terminal_bench.agents": MagicMock(),
            "terminal_bench.agents.terminus_2": MagicMock(
                Terminus2=mock_terminus_cls,
            ),
        },
    ):
        try:
            from ipw.agents.terminus import Terminus

            recorder = EventRecorder()
            agent = Terminus(model="gpt-4o", event_recorder=recorder)
            session = MagicMock()
            session.capture_pane.return_value = "terminal output"

            with patch.object(agent, "get_session", return_value=session):
                result = agent.run("solve this")

            assert result.content == "terminal output"
            events = recorder.get_events()
            event_types = [event.event_type for event in events]
            assert event_types.count(EventType.LM_INFERENCE_START) == 1
            assert event_types.count(EventType.LM_INFERENCE_END) == 1
            assert event_types.count(EventType.TOOL_CALL_START) == 1
            assert event_types.count(EventType.TOOL_CALL_END) == 1
            assert {
                event.metadata["tool"]
                for event in events
                if event.event_type in {EventType.TOOL_CALL_START, EventType.TOOL_CALL_END}
            } == {"terminal"}
        finally:
            AgentRegistry._entries().pop("terminus", None)
            sys.modules.pop("ipw.agents.terminus", None)
