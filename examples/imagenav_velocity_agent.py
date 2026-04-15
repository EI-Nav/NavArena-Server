#!/usr/bin/env python3
"""ImageNav velocity agent — ``eval_type: "imagenav"`` + ``action_space: "velocity"``.

Demonstrates how to access the ``goal_image`` observation that is unique to
ImageNav tasks, and returns random unicycle velocity commands.

Replace the random policy with your own visual-matching / learned model
to build a real ImageNav agent.

Pair with a bench config like ``configs/eval/imagenav_eval.yaml`` and set
``eval_settings.action_space: "velocity"``.

Usage:
    python examples/imagenav_velocity_agent.py --port 8000
"""

import argparse
import random

from navarena_server.server import NavigationModelServer, serve


class ImageNavVelocityAgent(NavigationModelServer):
    async def predict(self, observation, ctx):
        goal_image = observation.get("goal_image")
        if ctx.is_first and goal_image is not None:
            h, w = goal_image.shape[:2]
            print(f"  Received goal_image: {w}x{h}")

        # TODO: compare goal_image with current rgb to compute a heading error,
        #       then derive (v, w) from it.  The random baseline below is just
        #       a placeholder.
        return {
            "v": random.uniform(0.0, 1.0),
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
        description="NavArena ImageNav velocity agent"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(
        f"Starting ImageNav velocity agent on ws://{args.host}:{args.port}"
    )
    serve(ImageNavVelocityAgent(), host=args.host, port=args.port)
