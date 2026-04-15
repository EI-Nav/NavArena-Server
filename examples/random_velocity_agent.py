#!/usr/bin/env python3
"""Random velocity agent — ``action_space: "velocity"``.

Returns a random unicycle velocity command per decision step.
The bench integrates it via Euler integration:
``x = v * dt``, ``y = 0``, ``yaw = w * dt``.

Pair with ``eval_settings.action_space: "velocity"`` in the bench config.

Usage:
    python examples/random_velocity_agent.py --port 8000
"""

import argparse
import random

from navarena_server.server import NavigationModelServer, serve


class RandomVelocityAgent(NavigationModelServer):
    async def predict(self, observation, ctx):
        return {
            "v": random.uniform(-0.5, 1.0),
            "w": random.uniform(-1.0, 1.0),
            "dt": 0.1,
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
    parser = argparse.ArgumentParser(
        description="NavArena random velocity agent"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Starting random velocity agent on ws://{args.host}:{args.port}")
    serve(RandomVelocityAgent(), host=args.host, port=args.port)
