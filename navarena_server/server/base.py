"""Base NavigationModelServer ABC and SessionContext."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal


class SessionContext:
    def __init__(
        self,
        session_id: str,
        episode_id: str,
        mode: Literal["sync", "realtime"] = "sync",
        task: dict[str, Any] | None = None,
        slot_id: str | None = None,
    ) -> None:
        self._session_id = session_id
        self._episode_id = episode_id
        self._mode = mode
        self._step = 0
        self._task: dict[str, Any] = task or {}
        self._slot_id = slot_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def mode(self) -> Literal["sync", "realtime"]:
        return self._mode

    @property
    def step(self) -> int:
        return self._step

    @property
    def is_first(self) -> bool:
        return self._step == 0

    @property
    def task(self) -> dict[str, Any]:
        return self._task

    @property
    def slot_id(self) -> str | None:
        return self._slot_id

    def advance_step(self) -> None:
        """Advance the step counter by one."""
        self._step += 1

    def _increment_step(self) -> None:
        """Deprecated: use advance_step() instead."""
        self.advance_step()


class NavigationModelServer(ABC):
    @abstractmethod
    async def predict(
        self, observation: dict[str, Any], ctx: SessionContext
    ) -> dict[str, Any]:
        ...

    async def on_episode_start(  # noqa: B027  optional lifecycle hook
        self, payload: dict[str, Any], ctx: SessionContext
    ) -> None:
        """Called when an episode begins. Override to initialize episode state.

        ``payload`` is the full EPISODE_START message payload, typically
        ``{"task": {...}}`` where the ``task`` dict holds episode-level fields.
        """

    async def on_episode_end(  # noqa: B027  optional lifecycle hook
        self, result: dict[str, Any], ctx: SessionContext
    ) -> None:
        """Called when an episode ends. Override to clean up episode state."""

    async def batch_predict(
        self,
        observations: dict[str, dict[str, Any]],
        contexts: dict[str, SessionContext],
    ) -> dict[str, dict[str, Any]]:
        """Batch inference over multiple slots.

        The default implementation calls :meth:`predict` sequentially for
        each slot.  Override this method to perform true batched inference.
        """
        results: dict[str, dict[str, Any]] = {}
        for slot_id, obs in observations.items():
            results[slot_id] = await self.predict(obs, contexts[slot_id])
        return results

    async def on_batch_episode_start(  # noqa: B027
        self,
        payload: dict[str, Any],
        contexts: dict[str, SessionContext],
    ) -> None:
        """Called when a batch of episodes starts.

        Default delegates to :meth:`on_episode_start` per slot.
        """
        for slot_id, episode_data in payload.get("episodes", {}).items():
            await self.on_episode_start(episode_data, contexts[slot_id])

    async def on_batch_episode_end(  # noqa: B027
        self,
        payload: dict[str, Any],
        contexts: dict[str, SessionContext],
    ) -> None:
        """Called when some episodes in the batch end.

        Default delegates to :meth:`on_episode_end` per slot.
        """
        for slot_id, result in payload.get("episodes", {}).items():
            if slot_id in contexts:
                await self.on_episode_end(result, contexts[slot_id])
