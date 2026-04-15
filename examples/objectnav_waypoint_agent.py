#!/usr/bin/env python3
"""ObjectNav waypoint agent — ``eval_type: "objectnav"`` + ``action_space: "waypoint"``.

Demonstrates how to access the ``goal_category`` observation that is unique to
ObjectNav tasks, and returns random relative-pose waypoints.

The default success policy for ObjectNav is ``geometric_distance``, so the
benchmark auto-terminates the episode when the agent is within
``success_distance`` of the target object — no explicit STOP action is needed.

Replace the random policy with your own object-seeking logic (e.g. an object
detector or VLM) to build a real ObjectNav agent.

Pair with a bench config like ``configs/eval/objectnav_eval.yaml`` and set
``eval_settings.action_space: "waypoint"``.

Usage:
    python examples/objectnav_waypoint_agent.py --port 8000
"""

import argparse
import random

from navarena_server.server import NavigationModelServer, serve


class ObjectNavWaypointAgent(NavigationModelServer):
    async def predict(self, observation, ctx):
        goal_category = observation.get("goal_category")
        if ctx.is_first and goal_category is not None:
            print(f"  Target object: {goal_category}")

        # TODO: use goal_category with an object detector or VLM to locate
        #       the target object and steer toward it.
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
        description="NavArena ObjectNav waypoint agent"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(
        f"Starting ObjectNav waypoint agent on ws://{args.host}:{args.port}"
    )
    serve(ObjectNavWaypointAgent(), host=args.host, port=args.port)
