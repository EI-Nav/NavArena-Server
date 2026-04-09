#!/usr/bin/env python3
"""Reference WebSocket agent using navarena-server SDK.

Usage:
    python examples/random_agent.py --port 8000
"""

import argparse
import random

from navarena_server.server import NavigationModelServer, serve


class RandomNavigationAgent(NavigationModelServer):
    async def predict(self, observation, ctx):
        return {
            "x": random.uniform(-0.3, 0.5),
            "y": random.uniform(-0.1, 0.1),
            "yaw": random.uniform(-0.2, 0.2),
        }

    async def on_episode_start(self, task_info, ctx):
        print(
            f"[Episode {ctx.episode_id}] "
            f"Task: {ctx.task.get('task_type', 'unknown')}"
        )

    async def on_episode_end(self, result, ctx):
        print(
            f"[Episode {ctx.episode_id}] "
            f"Done: {result.get('success', 'unknown')} after {ctx.step} steps"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NavArena random agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Starting random agent server on ws://{args.host}:{args.port}")
    serve(RandomNavigationAgent(), host=args.host, port=args.port)
