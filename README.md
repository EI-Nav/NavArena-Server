# navarena-server

Lightweight WebSocket model-server SDK for [NavArena](https://github.com/EI-Nav/NavArena) agent developers.

## Install

```bash
pip install navarena-server
```

## Quick Start

The action format must match `eval_settings.action_space` in the bench
config.  Two action spaces are supported:

### Waypoint (default)

```python
from navarena_server.server import NavigationModelServer, serve

class MyAgent(NavigationModelServer):
    async def predict(self, observation, ctx):
        return {"waypoints": [{"x": 0.1, "y": 0.0, "yaw": 0.0}]}

serve(MyAgent(), host="0.0.0.0", port=8000)
```

### Velocity

```python
from navarena_server.server import NavigationModelServer, serve

class MyAgent(NavigationModelServer):
    async def predict(self, observation, ctx):
        return {"v": 0.5, "w": 0.1, "dt": 0.1}

serve(MyAgent(), host="0.0.0.0", port=8000)
```

See `examples/random_waypoint_agent.py` and `examples/random_velocity_agent.py`
for complete working examples.

## What This Package Provides

- `NavigationModelServer` — abstract base class for agent implementations
- `serve()` / `serve_async()` — WebSocket server runner
- `protocol` — message types, msgpack codecs, and client `Connection`
