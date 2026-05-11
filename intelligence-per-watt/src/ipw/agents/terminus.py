"""Terminus agent implementation for terminal-based tasks."""

from __future__ import annotations

import time
import types
from typing import TYPE_CHECKING, Any, Optional

from ipw.agents.base import BaseAgent
from ipw.core.registry import AgentRegistry
from ipw.core.types import AgentRunResult

if TYPE_CHECKING:
    from ipw.telemetry.events import EventRecorder

# Default Docker image with tmux pre-installed
DEFAULT_DOCKER_IMAGE = "ubuntu:22.04"


@AgentRegistry.register("terminus")
class Terminus(BaseAgent):
    """Terminus agent for terminal-based task execution in Docker containers."""

    DEFAULT_INSTRUCTIONS = (
        "You are a helpful assistant that can answer questions "
        "and use the tools provided to you if necessary."
    )

    def __init__(
        self,
        model: str,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        container_name: str | None = None,
        event_recorder: Optional["EventRecorder"] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Terminus agent.

        Args:
            model: The model name to use (e.g., "gpt-4o").
            docker_image: Docker image to use for the container. Must have tmux installed.
            container_name: Optional name for the Docker container.
            event_recorder: Optional EventRecorder for per-action energy telemetry.
            **kwargs: Additional keyword arguments passed to Terminus2.
        """
        super().__init__(event_recorder=event_recorder)

        # Lazy imports: docker and terminal-bench are optional
        try:
            import docker as _docker_mod  # noqa: F401
        except ImportError:
            raise ImportError(
                "docker package is required for Terminus agent. "
                "Install with: pip install docker"
            )

        try:
            from terminal_bench.agents.terminus_2 import Terminus2
        except ImportError:
            raise ImportError(
                "terminal-bench package is required for Terminus agent. "
                "Install with: pip install terminal-bench"
            )

        self.agent = Terminus2(model_name=model, **kwargs)
        self._records_lm_calls = self._instrument_lm_calls(model)
        self._docker_image = docker_image
        self._container_name = container_name or "terminus-container"
        self._docker_client = None
        self._container = None
        self._owns_container = False

    def _instrument_lm_calls(self, model_name: str) -> bool:
        """Wrap Terminus2's internal LLM call method for per-call events."""
        llm = getattr(self.agent, "_llm", None)
        if llm is None or getattr(llm, "_ipw_terminus_instrumented", False) is True:
            return False
        original_call = getattr(llm, "call", None)
        if not callable(original_call):
            return False

        def _estimate_tokens(_llm: Any, value: Any) -> int:
            try:
                return int(_llm.count_tokens(value))
            except Exception:
                return max(1, len(str(value)) // 4) if value is not None else 0

        def _instrumented_call(_llm: Any, *args: Any, **kwargs: Any) -> Any:
            prompt = kwargs.get("prompt")
            if prompt is None and args:
                prompt = args[0]
            message_history = kwargs.get("message_history") or []
            prompt_tokens = _estimate_tokens(_llm, message_history) + _estimate_tokens(
                _llm,
                prompt,
            )
            self._record_event(
                "lm_inference_start",
                model=model_name,
                prompt_tokens=prompt_tokens,
            )
            try:
                response = original_call(*args, **kwargs)
            except Exception as exc:
                self._record_event(
                    "lm_inference_end",
                    model=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    error=str(exc),
                )
                raise
            completion_tokens = _estimate_tokens(_llm, response)
            self._record_event(
                "lm_inference_end",
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return response

        llm.call = types.MethodType(_instrumented_call, llm)
        setattr(llm, "_ipw_terminus_instrumented", True)
        return True

    def _get_docker_client(self):
        """Get or create the Docker client."""
        if self._docker_client is None:
            import docker
            self._docker_client = docker.from_env()
        return self._docker_client

    def _get_or_create_container(self):
        """Get an existing container or create a new one with tmux installed."""
        import docker

        if self._container is not None:
            return self._container

        client = self._get_docker_client()

        # Try to get an existing container by name
        try:
            container = client.containers.get(self._container_name)
            if container.status != "running":
                container.start()
            self._container = container
            return container
        except docker.errors.NotFound:
            pass

        # Create a new container with tmux installed
        container = client.containers.run(
            self._docker_image,
            command="/bin/bash -c 'apt-get update && apt-get install -y tmux && tail -f /dev/null'",
            name=self._container_name,
            detach=True,
            tty=True,
            stdin_open=True,
        )
        self._container = container
        self._owns_container = True

        # Wait for tmux installation to complete
        for _ in range(30):
            exit_code, output = container.exec_run("which tmux")
            if exit_code == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Timeout waiting for tmux installation in container")

        return container

    def get_session(self, tmux_session: Any = None) -> Any:
        """Get or create a TmuxSession.

        Args:
            tmux_session: Either an existing TmuxSession, a session name string,
                or None to create a default session.

        Returns:
            A TmuxSession instance.
        """
        from terminal_bench.terminal.tmux_session import TmuxSession

        if isinstance(tmux_session, TmuxSession):
            return tmux_session

        container = self._get_or_create_container()
        session_name = tmux_session if isinstance(tmux_session, str) else "terminus-session"

        return TmuxSession(
            session_name=session_name,
            container=container,
            disable_recording=True,
        )

    def _instrument_session_tools(self, session: Any) -> Any:
        """Record terminal commands as tool calls on the provided session."""
        if getattr(session, "_ipw_tool_instrumented", False) is True:
            return session

        original_send_keys = session.send_keys

        def _send_keys_with_events(*args: Any, **kwargs: Any) -> Any:
            command = args[0] if args else kwargs.get("keys", "")
            command_text = str(command)
            self._record_event("tool_call_start", tool="terminal", command=command_text)
            try:
                return original_send_keys(*args, **kwargs)
            finally:
                self._record_event("tool_call_end", tool="terminal", command=command_text)

        session.send_keys = _send_keys_with_events
        setattr(session, "_ipw_tool_instrumented", True)
        return session

    def run(
        self,
        input: str,
        tmux_session: Any = None,
        **kwargs: Any,
    ) -> AgentRunResult:
        """Run the Terminus agent.

        Args:
            input: The input message or prompt for the agent.
            tmux_session: Optional TmuxSession or session name.
            **kwargs: Additional keyword arguments passed to agent.perform_task().

        Returns:
            AgentRunResult with the terminal output.
        """
        if not self._records_lm_calls:
            self._record_event("lm_inference_start", model=str(self.agent))
        try:
            session = self._instrument_session_tools(self.get_session(tmux_session))
            agent_result = self.agent.perform_task(input, session=session, **kwargs)

            terminal_output = session.capture_pane(capture_entire=True)
            return AgentRunResult(
                content=terminal_output,
                input_tokens=getattr(agent_result, "total_input_tokens", 0),
                output_tokens=getattr(agent_result, "total_output_tokens", 0),
            )
        finally:
            if not self._records_lm_calls:
                self._record_event("lm_inference_end", model=str(self.agent))

    def cleanup(self) -> None:
        """Clean up Docker resources."""
        if self._container is not None and self._owns_container:
            try:
                self._container.stop()
                self._container.remove()
            except Exception:
                pass
            self._container = None

    def __del__(self) -> None:
        """Destructor to clean up resources."""
        self.cleanup()
