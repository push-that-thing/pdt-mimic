"""Thin wrapper around `lerobot_record` that pre-imports our BYOP policy.

LeRobot v0.4.0 auto-discovers third-party `lerobot_robot_*`, `lerobot_camera_*`,
and `lerobot_teleoperator_*` plugins but NOT `lerobot_policy_*`. Importing the
package here registers `MimicVideoConfig` with draccus's ChoiceRegistry before
the CLI parses `--policy.type=mimic_video`.
"""
import lerobot_policy_mimic_video  # noqa: F401 — side-effect import: register policy type

from lerobot.scripts.lerobot_record import main

if __name__ == "__main__":
    main()
