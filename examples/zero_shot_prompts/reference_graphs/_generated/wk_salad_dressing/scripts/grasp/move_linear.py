"""Move the end-effector to a TCP-frame target along a straight Cartesian line.

Now drives motion through the connector's ``robot.go_to_pose_cartesian``,
which routes through :class:`gap.connector.ik.CuRoboBackend` and applies the
configured TCP offset / TCP rotation before planning. Earlier this script
called ``curobo.plan_linear`` (the bundle tool) directly, which treats the
input pose as the IK-link frame and is *not* TCP-aware — the workflow had to
encode the link-frame offset into its pose math.

``robot.go_to_pose_cartesian`` falls back internally to ``plan_to_pose``
(the v0.8 collision-aware single-pose planner) when the linear plan can't
solve, so this script still completes when clutter blocks the straight
line — same fallback semantics the previous version had.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Se3Pose


class Output(TypedDict):
    reached_pose: Se3Pose


def run(ctx: NodeContext, pose: Se3Pose) -> Output:
    ctx.tool("robot.go_to_pose_cartesian", pose=pose)
    return {"reached_pose": pose}
