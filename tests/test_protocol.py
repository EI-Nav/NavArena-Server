"""Protocol unit tests."""

import asyncio
import time

import anyio
import numpy as np
import pytest
import websockets

from navarena_server.protocol.messages import Message, MessageType, pack_message, unpack_message


def test_message_types_exist():
    from navarena_server.protocol.messages import MessageType

    assert MessageType.OBSERVATION == "observation"
    assert MessageType.ACTION == "action"
    assert MessageType.EPISODE_START == "episode_start"
    assert MessageType.EPISODE_END == "episode_end"
    assert MessageType.ERROR == "error"


def test_pack_unpack_roundtrip():
    from navarena_server.protocol.messages import (
        Message,
        MessageType,
        pack_message,
        unpack_message,
    )

    msg = Message(
        type=MessageType.ACTION,
        payload={"x": 0.5, "y": 0.0, "yaw": 0.1},
        seq=42,
    )
    data = pack_message(msg)
    assert isinstance(data, bytes)

    restored = unpack_message(data)
    assert restored.type == MessageType.ACTION
    assert restored.payload == {"x": 0.5, "y": 0.0, "yaw": 0.1}
    assert restored.seq == 42
    assert isinstance(restored.timestamp, float)


def test_unpack_rejects_malformed_data():
    from navarena_server.protocol.messages import unpack_message

    with pytest.raises(ValueError, match="Failed to decode"):
        unpack_message(b"not-msgpack")


def test_unpack_rejects_missing_fields():
    import msgpack

    from navarena_server.protocol.messages import unpack_message

    data = msgpack.packb({"type": "action"})
    with pytest.raises(ValueError, match="missing required"):
        unpack_message(data)


def test_unpack_rejects_unknown_type():
    import msgpack

    from navarena_server.protocol.messages import unpack_message

    data = msgpack.packb({
        "type": "unknown",
        "payload": {},
        "seq": 0,
        "timestamp": time.time(),
    })
    with pytest.raises(ValueError, match="Unknown message type"):
        unpack_message(data)


# ---------- numpy codec tests ----------


def test_numpy_array_roundtrip():
    from navarena_server.protocol.numpy_codec import decode_ndarray, encode_ndarray

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    encoded = encode_ndarray(arr)
    assert isinstance(encoded, dict)
    assert encoded["__ndarray__"] is True

    decoded = decode_ndarray(encoded)
    np.testing.assert_array_equal(decoded, arr)
    assert decoded.dtype == np.float32


def test_numpy_scalar_types_converted():
    from navarena_server.protocol.numpy_codec import encode_ndarray

    assert isinstance(encode_ndarray(np.int64(42)), int)
    assert isinstance(encode_ndarray(np.float32(3.14)), float)
    assert isinstance(encode_ndarray(np.bool_(True)), bool)


def test_numpy_rejects_unsafe_dtypes():
    from navarena_server.protocol.numpy_codec import decode_ndarray

    with pytest.raises(ValueError, match="Disallowed numpy dtype"):
        decode_ndarray({
            "__ndarray__": True,
            "data": b"\x00",
            "dtype": "O",
            "shape": [1],
        })


def test_message_roundtrip_with_numpy():
    from navarena_server.protocol.messages import (
        Message,
        MessageType,
        pack_message,
        unpack_message,
    )

    arr = np.random.rand(64, 64, 3).astype(np.float32)
    msg = Message(
        type=MessageType.OBSERVATION,
        payload={"image": arr, "scalar": np.float32(1.5)},
        seq=1,
    )
    data = pack_message(msg)
    restored = unpack_message(data)

    np.testing.assert_array_almost_equal(restored.payload["image"], arr)
    assert restored.payload["scalar"] == pytest.approx(1.5)


# ---------- image codec tests ----------


def test_image_jpeg_roundtrip():
    from navarena_server.protocol.image_codec import decode_image, encode_image, is_encoded_image

    img = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
    encoded = encode_image(img, "jpeg")
    assert is_encoded_image(encoded)
    assert encoded["format"] == "jpeg"

    decoded = decode_image(encoded)
    assert decoded.shape == img.shape
    assert decoded.dtype == np.uint8


def test_image_png_roundtrip_lossless():
    from navarena_server.protocol.image_codec import decode_image, encode_image

    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    encoded = encode_image(img, "png")
    decoded = decode_image(encoded)
    np.testing.assert_array_equal(decoded, img)


def test_image_raw_roundtrip():
    from navarena_server.protocol.image_codec import decode_image, encode_image

    img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    encoded = encode_image(img, "raw")
    decoded = decode_image(encoded)
    np.testing.assert_array_equal(decoded, img)


def test_numpy_codec_uses_image_format_for_hwc_uint8():
    from navarena_server.protocol.numpy_codec import (
        decode_ndarray,
        encode_ndarray,
        get_image_format,
        set_image_format,
    )

    original_fmt = get_image_format()
    try:
        set_image_format("png")
        img = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
        encoded = encode_ndarray(img)
        assert "__image__" in encoded

        decoded = decode_ndarray(encoded)
        np.testing.assert_array_equal(decoded, img)
    finally:
        set_image_format(original_fmt)


# ---------- connection tests ----------


@pytest.fixture
def echo_server():
    """Start a local WebSocket echo server in a background thread."""
    import threading

    async def handler(ws):
        async for data in ws:
            msg = unpack_message(data)
            if msg.type == MessageType.OBSERVATION:
                response = Message(
                    type=MessageType.ACTION,
                    payload={"x": 0.1, "y": 0.0, "yaw": 0.0},
                    seq=msg.seq,
                )
                await ws.send(pack_message(response))

    port_holder: list[int] = []
    ready = threading.Event()
    stop = threading.Event()

    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def serve():
            server = await websockets.serve(handler, "127.0.0.1", 0, compression=None)
            port_holder.append(server.sockets[0].getsockname()[1])
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.05)
            server.close()
            await server.wait_closed()

        loop.run_until_complete(serve())
        loop.close()

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    ready.wait(timeout=5)
    yield f"ws://127.0.0.1:{port_holder[0]}"
    stop.set()
    t.join(timeout=5)


def test_connection_act(echo_server):
    from navarena_server.protocol.connection import Connection

    async def run():
        async with Connection(echo_server, timeout=5.0) as conn:
            action = await conn.act({"rgb": {}, "pose": {}})
            assert action == {"x": 0.1, "y": 0.0, "yaw": 0.0}

    anyio.run(run)


def test_connection_episode_lifecycle(echo_server):
    from navarena_server.protocol.connection import Connection

    async def run():
        async with Connection(echo_server, timeout=5.0) as conn:
            await conn.start_episode({"task": {"task_type": "pointnav"}})
            action = await conn.act({"rgb": {}})
            assert "x" in action
            await conn.end_episode({"success": True})

    anyio.run(run)


# ---------- model server integration tests ----------


def test_model_server_integration():
    """End-to-end: Connection client <-> NavigationModelServer."""
    import threading

    from navarena_server.protocol.connection import Connection
    from navarena_server.server import NavigationModelServer
    from navarena_server.server.serve import _handle_connection

    class FixedAgent(NavigationModelServer):
        async def predict(self, observation, ctx):
            return {"x": 0.25, "y": 0.0, "yaw": 0.05}

    agent = FixedAgent()
    port_holder: list[int] = []
    ready = threading.Event()
    stop = threading.Event()

    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def serve():
            async def handler(ws):
                await _handle_connection(ws, agent)

            server = await websockets.serve(
                handler, "127.0.0.1", 0, compression=None, max_size=None, ping_interval=None,
            )
            port_holder.append(server.sockets[0].getsockname()[1])
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.05)
            server.close()
            await server.wait_closed()

        loop.run_until_complete(serve())
        loop.close()

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    ready.wait(timeout=5)

    async def run():
        async with Connection(f"ws://127.0.0.1:{port_holder[0]}", timeout=5.0) as conn:
            await conn.start_episode({"task": {"task_type": "pointnav"}})
            action = await conn.act({"rgb": {}, "pose": {"position": [0, 0, 0]}})
            assert action == {"x": 0.25, "y": 0.0, "yaw": 0.05}
            await conn.end_episode({"success": True})

    try:
        anyio.run(run)
    finally:
        stop.set()
        t.join(timeout=5)
