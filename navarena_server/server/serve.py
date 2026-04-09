"""WebSocket server runner for NavigationModelServer instances."""

from __future__ import annotations

import logging
import uuid
from functools import partial
from typing import Any

import anyio
import websockets

from navarena_server.protocol.messages import (
    Message,
    MessageType,
    pack_message,
    unpack_message,
)
from navarena_server.server.base import NavigationModelServer, SessionContext

logger = logging.getLogger(__name__)


async def _handle_connection(ws: Any, model_server: NavigationModelServer) -> None:
    session_id = str(uuid.uuid4())
    ctx = SessionContext(session_id=session_id, episode_id="")
    current_obs_seq: list[int] = [0]

    # Per-slot contexts for batch mode
    slot_contexts: dict[str, SessionContext] = {}
    batch_mode = False

    async def send_action(action: dict[str, Any]) -> None:
        msg = Message(
            type=MessageType.ACTION, payload=action, seq=current_obs_seq[0]
        )
        await ws.send(pack_message(msg))

    in_episode = False

    try:
        async for raw_data in ws:
            msg = unpack_message(raw_data)

            if msg.type == MessageType.EPISODE_START:
                task = msg.payload.get("task", {})
                episode_id = (
                    str(task.get("episode_id"))
                    if task.get("episode_id") is not None
                    else str(uuid.uuid4())
                )
                ctx = SessionContext(
                    session_id=session_id,
                    episode_id=episode_id,
                    task=task,
                )
                try:
                    await model_server.on_episode_start(msg.payload, ctx)
                    in_episode = True
                except Exception as exc:
                    logger.exception("Error in on_episode_start")
                    error_msg = Message(
                        type=MessageType.ERROR,
                        payload={"error": str(exc)},
                        seq=msg.seq,
                    )
                    await ws.send(pack_message(error_msg))

            elif msg.type == MessageType.OBSERVATION:
                if not in_episode:
                    logger.warning("Received OBSERVATION outside of episode")
                    error_msg = Message(
                        type=MessageType.ERROR,
                        payload={
                            "error": "Received OBSERVATION outside of an active episode. "
                            "Send EPISODE_START first.",
                        },
                        seq=msg.seq,
                    )
                    await ws.send(pack_message(error_msg))
                    continue
                current_obs_seq[0] = msg.seq
                try:
                    action = await model_server.predict(msg.payload, ctx)
                    await send_action(action)
                except Exception as exc:
                    logger.exception("Error in predict")
                    in_episode = False
                    error_msg = Message(
                        type=MessageType.ERROR,
                        payload={"error": str(exc), "fatal": True},
                        seq=msg.seq,
                    )
                    await ws.send(pack_message(error_msg))
                    continue
                ctx.advance_step()

            elif msg.type == MessageType.EPISODE_END:
                try:
                    await model_server.on_episode_end(msg.payload, ctx)
                except Exception:
                    logger.exception("Error in on_episode_end")
                in_episode = False

            elif msg.type == MessageType.BATCH_EPISODE_START:
                batch_mode = True
                episodes = msg.payload.get("episodes", {})
                for slot_id, ep_data in episodes.items():
                    task = ep_data.get("task", {})
                    episode_id = (
                        str(task.get("episode_id"))
                        if task.get("episode_id") is not None
                        else str(uuid.uuid4())
                    )
                    slot_contexts[slot_id] = SessionContext(
                        session_id=session_id,
                        episode_id=episode_id,
                        task=task,
                        slot_id=slot_id,
                    )
                try:
                    await model_server.on_batch_episode_start(
                        msg.payload, slot_contexts
                    )
                    in_episode = True
                except Exception as exc:
                    logger.exception("Error in on_batch_episode_start")
                    error_msg = Message(
                        type=MessageType.ERROR,
                        payload={"error": str(exc)},
                        seq=msg.seq,
                    )
                    await ws.send(pack_message(error_msg))

            elif msg.type == MessageType.BATCH_OBSERVATION:
                if not batch_mode:
                    logger.warning(
                        "Received BATCH_OBSERVATION without BATCH_EPISODE_START"
                    )
                    error_msg = Message(
                        type=MessageType.ERROR,
                        payload={
                            "error": "Received BATCH_OBSERVATION without an active batch. "
                            "Send BATCH_EPISODE_START first.",
                        },
                        seq=msg.seq,
                    )
                    await ws.send(pack_message(error_msg))
                    continue
                current_obs_seq[0] = msg.seq
                observations = msg.payload.get("observations", {})
                try:
                    actions = await model_server.batch_predict(
                        observations, slot_contexts
                    )
                    response = Message(
                        type=MessageType.BATCH_ACTION,
                        payload={"actions": actions},
                        seq=msg.seq,
                    )
                    await ws.send(pack_message(response))
                except Exception as exc:
                    logger.exception("Error in batch_predict")
                    error_msg = Message(
                        type=MessageType.ERROR,
                        payload={"error": str(exc), "fatal": True},
                        seq=msg.seq,
                    )
                    await ws.send(pack_message(error_msg))
                    continue
                for slot_id in actions:
                    if slot_id in slot_contexts:
                        slot_contexts[slot_id].advance_step()

            elif msg.type == MessageType.BATCH_EPISODE_END:
                ended = msg.payload.get("episodes", {})
                try:
                    await model_server.on_batch_episode_end(
                        msg.payload, slot_contexts
                    )
                except Exception:
                    logger.exception("Error in on_batch_episode_end")
                for slot_id in ended:
                    slot_contexts.pop(slot_id, None)
                if not slot_contexts:
                    batch_mode = False
                    in_episode = False

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected: session=%s", session_id[:8])
    except Exception:
        logger.exception("Error handling session=%s", session_id[:8])
    finally:
        if in_episode and not batch_mode:
            try:
                await model_server.on_episode_end({}, ctx)
            except Exception:
                logger.exception("Cleanup on_episode_end failed")
        elif batch_mode and slot_contexts:
            for slot_id, s_ctx in list(slot_contexts.items()):
                try:
                    await model_server.on_episode_end({}, s_ctx)
                except Exception:
                    logger.exception(
                        "Cleanup on_episode_end failed for slot %s", slot_id
                    )


async def serve_async(
    model_server: NavigationModelServer,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    logger.info("Starting model server on ws://%s:%d", host, port)

    async def handler(ws: Any) -> None:
        await _handle_connection(ws, model_server)

    async with websockets.serve(
        handler, host, port, compression=None, max_size=None, ping_interval=None
    ):
        await anyio.sleep_forever()


def serve(
    model_server: NavigationModelServer,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    anyio.run(partial(serve_async, model_server, host, port))
