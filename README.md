# navarena-server

Lightweight WebSocket model-server SDK for [NavArena](https://github.com/EI-Nav/NavArena) agent developers.

## Install

```bash
pip install navarena-server
```

## Quick Start

```python
from navarena_server.server import NavigationModelServer, serve

class MyAgent(NavigationModelServer):
    async def predict(self, observation, ctx):
        return {"x": 0.1, "y": 0.0, "yaw": 0.0}

serve(MyAgent(), host="0.0.0.0", port=8000)
```

## What This Package Provides

- `NavigationModelServer` — abstract base class for agent implementations
- `serve()` / `serve_async()` — WebSocket server runner
- `protocol` — message types, msgpack codecs, and client `Connection`
