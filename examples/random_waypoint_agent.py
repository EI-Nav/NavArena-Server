#!/usr/bin/env python3
"""Random waypoint agent — ``action_space: "waypoint"`` (default).

Returns one random relative-pose waypoint per decision step.
Pair with ``eval_settings.action_space: "waypoint"`` in the bench config.

Usage:
    python examples/random_waypoint_agent.py --port 8000
"""

import argparse
import random

from navarena_server.server import NavigationModelServer, serve


class RandomWaypointAgent(NavigationModelServer):
    async def predict(self, observation, ctx):
        return {
            "waypoints": [
                {
                    "x": random.uniform(-0.3, 0.5),
                    "y": random.uniform(-0.1, 0.1),
                    "yaw": random.uniform(-0.2, 0.2),
                }
            ]
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
        description="NavArena random waypoint agent"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Starting random waypoint agent on ws://{args.host}:{args.port}")
    serve(RandomWaypointAgent(), host=args.host, port=args.port)
