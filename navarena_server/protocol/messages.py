"""Message types and serialization for the WebSocket protocol."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any

import msgpack

from navarena_server.protocol.numpy_codec import decode_ndarray, encode_ndarray


class MessageType(str, enum.Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    ERROR = "error"
    BATCH_OBSERVATION = "batch_observation"
    BATCH_ACTION = "batch_action"
    BATCH_EPISODE_START = "batch_episode_start"
    BATCH_EPISODE_END = "batch_episode_end"


@dataclass
class Message:
    type: MessageType
    payload: dict[str, Any]
    seq: int = 0
    timestamp: float = field(default_factory=time.time)


def pack_message(msg: Message) -> bytes:
    raw = {
        "type": msg.type.value,
        "payload": msg.payload,
        "seq": msg.seq,
        "timestamp": msg.timestamp,
    }
    return msgpack.packb(raw, default=encode_ndarray, use_bin_type=True)


def unpack_message(data: bytes) -> Message:
    try:
        raw = msgpack.unpackb(data, object_hook=decode_ndarray, raw=False)
    except Exception as exc:
        raise ValueError(
            f"Failed to decode msgpack data ({len(data)} bytes): {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Expected msgpack dict, got {type(raw).__name__}")

    _REQUIRED = ("type", "payload", "seq", "timestamp")
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise ValueError(
            f"Message missing required fields: {missing}. Got keys: {list(raw.keys())}"
        )

    try:
        msg_type = MessageType(raw["type"])
    except ValueError:
        raise ValueError(
            f"Unknown message type: {raw['type']!r}. "
            f"Valid types: {[t.value for t in MessageType]}"
        ) from None

    payload = raw["payload"]
    if not isinstance(payload, dict):
        raise ValueError(
            f"Message payload must be dict, got {type(payload).__name__}"
        )

    seq = raw["seq"]
    if not isinstance(seq, int):
        raise ValueError(
            f"Message seq must be int, got {type(seq).__name__}"
        )

    return Message(
        type=msg_type,
        payload=payload,
        seq=seq,
        timestamp=raw["timestamp"],
    )
