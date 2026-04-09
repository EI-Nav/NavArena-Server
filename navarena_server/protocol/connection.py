"""Connection client: communicates with ModelServer over WebSocket."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import anyio
import websockets
from websockets.protocol import State as WebSocketState

from navarena_server.protocol.messages import (
    Message,
    MessageType,
    pack_message,
    unpack_message,
)

logger = logging.getLogger(__name__)


class Connection:
    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        strict_seq: bool = True,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.strict_seq = strict_seq
        self._ws: Any = None
        self._seq: int = 0
        self._action_callback: Callable | None = None
        self._listener_task: asyncio.Task | None = None
        self._in_episode: bool = False

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ws.state is WebSocketState.OPEN

    async def connect(self) -> None:
        await self._connect_with_backoff()

    async def close(self) -> None:
        await self.stop_listener()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def send(self, msg_type: MessageType, payload: dict[str, Any]) -> int:
        seq = self._next_seq()
        msg = Message(type=msg_type, payload=payload, seq=seq)
        await self._ensure_connected()
        await self._ws.send(pack_message(msg))
        return seq

    async def recv(self, *, timeout: float | None = None) -> Message:
        if self._ws is None:
            raise RuntimeError("Not connected")
        with anyio.fail_after(timeout):
            data = await self._ws.recv()
        return unpack_message(data)

    async def start_episode(self, config: dict[str, Any]) -> None:
        await self.send(MessageType.EPISODE_START, config)
        self._in_episode = True

    async def end_episode(self, result: dict[str, Any]) -> None:
        await self.send(MessageType.EPISODE_END, result)
        self._in_episode = False

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        seq = await self.send(MessageType.OBSERVATION, obs)
        response = await self.recv(timeout=self.timeout)
        if response.type == MessageType.ERROR:
            raise RuntimeError(f"Server error: {response.payload}")
        if response.type != MessageType.ACTION:
            raise RuntimeError(
                f"Expected ACTION response, got {response.type.name}"
            )
        if response.seq != seq:
            if self.strict_seq:
                raise RuntimeError(
                    f"Seq mismatch: sent {seq}, received {response.seq}"
                )
            logger.warning("Seq mismatch: sent %d, received %d", seq, response.seq)
        return response.payload

    async def batch_act(self, observations: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Send N observations keyed by slot ID, receive N actions."""
        seq = await self.send(
            MessageType.BATCH_OBSERVATION, {"observations": observations}
        )
        response = await self.recv(timeout=self.timeout)
        if response.type == MessageType.ERROR:
            raise RuntimeError(f"Server error: {response.payload}")
        if response.type != MessageType.BATCH_ACTION:
            raise RuntimeError(
                f"Expected BATCH_ACTION response, got {response.type.name}"
            )
        if response.seq != seq:
            if self.strict_seq:
                raise RuntimeError(
                    f"Seq mismatch: sent {seq}, received {response.seq}"
                )
            logger.warning("Seq mismatch: sent %d, received %d", seq, response.seq)
        return response.payload.get("actions", {})

    async def batch_start_episodes(self, episodes: dict[str, dict[str, Any]]) -> None:
        """Notify the server that a batch of episodes is starting."""
        await self.send(
            MessageType.BATCH_EPISODE_START, {"episodes": episodes}
        )
        self._in_episode = True

    async def batch_end_episodes(self, results: dict[str, dict[str, Any]]) -> None:
        """Notify the server that some episodes in the batch have ended."""
        await self.send(
            MessageType.BATCH_EPISODE_END, {"episodes": results}
        )

    def batch_complete(self) -> None:
        """Clear episode tracking after all batch episodes have finished."""
        self._in_episode = False

    async def send_observation(self, obs: dict[str, Any]) -> None:
        await self.send(MessageType.OBSERVATION, obs)

    def on_action(self, callback: Callable) -> None:
        self._action_callback = callback

    async def start_listener(self) -> None:
        if self._listener_task is not None and not self._listener_task.done():
            return
        self._listener_task = asyncio.create_task(self._listener_loop())

    async def stop_listener(self) -> None:
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, anyio.get_cancelled_exc_class()):
                pass
            self._listener_task = None

    async def reconnect(self) -> None:
        await self.close()
        await self._connect_with_backoff()

    async def _connect_with_backoff(self) -> None:
        for attempt in range(1, self.max_retries + 1):
            try:
                with anyio.fail_after(self.timeout):
                    self._ws = await websockets.connect(
                        self.url,
                        compression=None,
                        max_size=None,
                        ping_interval=None,
                    )
                logger.info(
                    "Connected to %s (attempt %d/%d)",
                    self.url,
                    attempt,
                    self.max_retries,
                )
                return
            except anyio.get_cancelled_exc_class():
                raise
            except Exception as e:
                if attempt == self.max_retries:
                    raise ConnectionError(
                        f"Server unreachable at {self.url} after {self.max_retries} retries"
                    ) from e
                wait = self.backoff_base ** attempt
                logger.warning(
                    "Reconnect attempt %d/%d failed (%s), retrying in %.1fs...",
                    attempt,
                    self.max_retries,
                    type(e).__name__,
                    wait,
                )
                await anyio.sleep(wait)

    async def _ensure_connected(self) -> None:
        if self.is_connected:
            return
        if self._in_episode:
            self._in_episode = False
            raise ConnectionError(
                "Connection lost during active episode; the model server has "
                "lost episode state and cannot resume."
            )
        logger.warning("Connection lost, attempting reconnect...")
        await self._connect_with_backoff()

    async def _listener_loop(self) -> None:
        try:
            while True:
                msg = await self.recv()
                if msg.type == MessageType.ACTION and self._action_callback is not None:
                    self._action_callback(msg.payload)
        except (asyncio.CancelledError, anyio.get_cancelled_exc_class()):
            pass
        except Exception:
            logger.exception("Listener loop error")

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def __aenter__(self) -> Connection:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
