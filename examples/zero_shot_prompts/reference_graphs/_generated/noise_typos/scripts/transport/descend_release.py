"""Descend to the drop position, release the object, and retract home.

Three-step sequence: a straight-line Cartesian descend to the drop
position, open the gripper with settle delays, then go_home to retract.
This combines descend + release + retract into a single atomic node.

The descend uses ``robot.go_to_pose_cartesian`` (a TCP-aware straight
Cartesian line, backed by cuRobo's linear planner with a single-pose
fallback) rather than the free-space ``robot.go_to_pose``: a straight
vertical descend lowers the held object cleanly instead of the planner's
shortest-path arc that can swing or toss it just before release.

The gripper open uses ``settle_steps`` to keep the arm holding its target
while the sim steps, so contact-rich events (finger opening, object
falling) get physics resolution rather than the motion-to-motion yank that
caused visible "tossing" of the released object. ``time.sleep()`` does NOT
advance the sim (it only steps when a motion or hold command is in
flight); use ``settle_steps`` instead. The arm motions themselves
(``robot.go_to_pose_cartesian`` / ``robot.go_home``) block until the
controller has converged, which provides the descend/retract settling.
"""

from typing import TypedDict

from gap import NodeContext
from gap_core.types import Quaternion, Se3Pose, Vec3


class Output(TypedDict):
    drop_position: Vec3


def run(
    ctx: NodeContext,
    drop_position: Vec3,
    drop_rotation: Quaternion | None = None,
) -> Output:
    # Use the upstream-supplied drop rotation when available; fall back
    # to the canonical top-down quaternion only when the caller didn't
    # plumb a rotation through. Hardcoding ``_DOWN`` here forces the
    # wrist to unspool any preserved yaw mid-descent — visible as a
    # sudden swing that tosses the held object off-target. Pull the
    # rotation from ``compute_drop.drop_pose.rotation`` (yaw-only by
    # default; see ``compute_drop_pose._yaw_only_topdown``) and the
    # descend stays purely vertical.
    rotation: Quaternion = (
        drop_rotation
        if drop_rotation is not None
        else {"w": 0.0, "x": 1.0, "y": 0.0, "z": 0.0}
    )

    # Descend to the drop position along a straight Cartesian line so the
    # held object moves vertically down; the motion blocks until the arm is
    # stable, so the gripper opens from a quiescent pose.
    pose: Se3Pose = {"position": drop_position, "rotation": rotation}
    ctx.tool("robot.go_to_pose_cartesian", pose=pose)
    # Open + hold so the just-released object lands and settles in the
    # container BEFORE the arm starts retracting.
    ctx.tool("robot.open_gripper", settle_steps=60)
    # Retract slowly enough that the freshly-placed object isn't disturbed
    # by the arm's lateral motion.
    ctx.tool("robot.go_home")

    return {"drop_position": drop_position}
