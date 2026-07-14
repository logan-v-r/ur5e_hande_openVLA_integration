#!/usr/bin/env python3
"""
mirror_relative.py
=========
Standalone leader/follower mirroring process for teleoperated demo collection.

The LEADER (UR7e + robotiq HAND-E, hand-guided in freedrive) is continuously tracked by the 
FOLLOWER (UR5e + robotiq HAND-E). The follower moves relative to its own starting pose, based 
on the leader's motion relative to its own starting pose. The follower gripper 
tracks the leader gripper position.

--------------------------------------------------------------------------------
DEPENDENCIES:  pip install ur-rtde numpy scipy   (+ robotiq_gripper.py importable)
--------------------------------------------------------------------------------
SAFETY
    The FOLLOWER MOVES continuously to chase the leader. Keep an e-stop in hand.
    Start with both arms in similar, collision-free poses. Begin with a SLOW
    leader; the follower servos at low speed by default. A reachability guard
    holds the follower if the leader goes beyond a set distance.
--------------------------------------------------------------------------------
Execution:
    python mirror_relative.py --no-gripper       # arm pose only
    python mirror_relative.py                     # arm + leader-gripper mirroring
    python mirror_relative.py --keyboard-gripper  # arm + manual follower gripper

When --keyboard-gripper is enabled, press 'g' in this terminal to toggle the
UR5e gripper between its calibrated open and closed positions.
--------------------------------------------------------------------------------
"""

import argparse
import json
import os
import select
import signal
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface

import robotiq_gripper


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    LEADER_IP = "192.168.1.106"      # <-- SET: UR7e (leader) IP
    FOLLOWER_IP = "192.168.1.102"  # <-- SET: UR5e (follower) IP

    # Gripper: IP == the robot it's attached to; port fixed at 63352.
    GRIPPER_PORT = 63352
    LEADER_GRIPPER_IP = LEADER_IP
    FOLLOWER_GRIPPER_IP = FOLLOWER_IP
    LEADER_HAS_GRIPPER = True          # False -> follower gripper is not mirrored

    # --- Motion-frame transform: leader base orientation -> follower base orientation ---
    # For same-direction mounting, use None. Current value reflects current demo setup.
    FRAME_TRANSFORM = R.from_euler("z", 180, degrees=True)

    # --- Mirror loop rate ---
    RATE_HZ = 125.0                    # e-series RTDE control runs at 125-500 Hz;
                                       # 125 Hz is a safe, smooth default

    # --- servoL parameters (LOW speed for safety at first) ---
    SERVO_SPEED = 0.10                 # m/s
    SERVO_ACC = 0.50                   # m/s^2
    SERVO_LOOKAHEAD = 0.1              # s (0.03-0.2 valid)
    SERVO_GAIN = 300                   # (100-2000)

    # --- Tracking / unexpected-jump guard ---
    # With relative mirroring, this measures follower lag from the intended
    # relative-motion target. It is NOT a complete UR5e workspace check.
    # Start conservatively; increase only after verifying smooth, safe motion.
    MAX_CHASE_DIST_M = 0.20

    # --- Gripper mirroring ---
    GRIP_SPEED = 150                   # 0..255
    GRIP_FORCE = 100                   # 0..255 (moderate; raise for firmer grip)
    GRIP_DEADBAND = 3                  # counts; ignore tiny leader changes
    GRIP_UPDATE_HZ = 20.0              # gripper relay rate (slower than arm loop)


# =============================================================================
# helpers
# =============================================================================
def apply_frame_transform(leader_pose, transform):
    """Map a leader base-frame TCP pose into the follower base frame."""
    if transform is None:
        return list(leader_pose)
    p = np.asarray(leader_pose, float)
    new_pos = transform.apply(p[:3])
    new_rot = (transform * R.from_rotvec(p[3:6])).as_rotvec()
    return np.concatenate([new_pos, new_rot]).tolist()


def dist(pose_a, pose_b):
    return float(np.linalg.norm(np.asarray(pose_a[:3]) - np.asarray(pose_b[:3])))


def relative_pose_target(leader_start, leader_current, follower_start, transform=None):
    """Map leader motion since startup onto the follower's startup TCP pose.

    Unlike absolute TCP-pose mirroring, this does not assume the two robot
    bases share the same physical origin.  The follower remains at its own
    startup pose until the leader is hand-guided away from its startup pose.

    ``transform`` is optional and is only needed to rotate leader motion into
    the follower base orientation (for example, a 180-degree Z rotation when
    the robots face each other).  For same-direction mounting, use ``None``.
    """
    leader_start = np.asarray(leader_start, dtype=float)
    leader_current = np.asarray(leader_current, dtype=float)
    follower_start = np.asarray(follower_start, dtype=float)

    # Translation made by the leader since startup, expressed in leader base.
    delta_xyz = leader_current[:3] - leader_start[:3]
    if transform is not None:
        delta_xyz = transform.apply(delta_xyz)
    target_xyz = follower_start[:3] + delta_xyz

    # Orientation change made by the leader since startup.
    r_leader_start = R.from_rotvec(leader_start[3:6])
    r_leader_current = R.from_rotvec(leader_current[3:6])
    delta_rot = r_leader_current * r_leader_start.inv()

    # Rotate that orientation change into follower-base convention, if needed.
    if transform is not None:
        delta_rot = transform * delta_rot * transform.inv()

    r_follower_start = R.from_rotvec(follower_start[3:6])
    target_rot = delta_rot * r_follower_start

    return np.concatenate([target_xyz, target_rot.as_rotvec()]).tolist()


def connect_gripper(ip, port):
    g = robotiq_gripper.RobotiqGripper()
    print(f"  connecting gripper at {ip}:{port} ...")
    g.connect(ip, port)
    if not g.is_active():
        print("  activating gripper (this moves it) ...")
        g.activate()
        time.sleep(2)
    return g


class TerminalKeyReader:
    """Read single key presses from the mirror terminal without blocking motion.

    This helper is used only for the optional manual follower-gripper toggle.
    It places the terminal in cbreak mode while mirroring and restores the
    original terminal settings during shutdown.
    """

    def __init__(self):
        self.enabled = False
        self._fd = None
        self._original_settings = None

    def start(self):
        """Enable non-blocking single-key input when running in an interactive terminal."""
        if not sys.stdin.isatty():
            print("  [gripper] keyboard control unavailable: standard input is not a terminal.")
            return

        self._fd = sys.stdin.fileno()
        self._original_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.enabled = True

    def read_keys(self):
        """Return all keys currently waiting in the terminal input buffer."""
        if not self.enabled:
            return []

        keys = []
        while select.select([sys.stdin], [], [], 0)[0]:
            keys.append(sys.stdin.read(1))
        return keys

    def close(self):
        """Restore normal terminal input settings after mirroring stops."""
        if self.enabled and self._fd is not None and self._original_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_settings)
        self.enabled = False


# =============================================================================
# MIRROR
# =============================================================================
class Mirror:
    def __init__(
        self,
        cfg: Config,
        mirror_gripper: bool,
        keyboard_gripper: bool,
        gripper_status_file: Path | None,
    ):
        """Connect the robots and configure either mirrored or keyboard gripper control."""
        self.cfg = cfg
        self.mirror_gripper = mirror_gripper and cfg.LEADER_HAS_GRIPPER
        self.keyboard_gripper = keyboard_gripper
        self.gripper_status_file = gripper_status_file.expanduser() if gripper_status_file else None

        print("Connecting to robots...")
        self.rtde_leader = RTDEReceiveInterface(cfg.LEADER_IP)
        self.rtde_follower = RTDEReceiveInterface(cfg.FOLLOWER_IP)
        # Control interface on BOTH arms: follower for servoL, leader for
        # teachMode() (freedrive hand-guiding). The leader control iface is only
        # used to enter/exit teach mode -- the mirror never commands leader motion.
        self.ctrl_leader = RTDEControlInterface(cfg.LEADER_IP)
        self.ctrl_follower = RTDEControlInterface(cfg.FOLLOWER_IP)

        self.lead_g = None
        self.foll_g = None
        if self.mirror_gripper:
            self.foll_g = connect_gripper(cfg.FOLLOWER_GRIPPER_IP, cfg.GRIPPER_PORT)
            self.lead_g = connect_gripper(cfg.LEADER_GRIPPER_IP, cfg.GRIPPER_PORT)
        elif self.keyboard_gripper:
            # Keyboard mode needs only the follower gripper connection.
            self.foll_g = connect_gripper(cfg.FOLLOWER_GRIPPER_IP, cfg.GRIPPER_PORT)

        self._stop = False
        self._teach_active = False        # track teach mode for guaranteed cleanup
        self._last_grip_cmd = None
        self._last_grip_update = 0.0
        self._manual_gripper_closed = False
        self._manual_gripper_target = None
        self._keyboard = TerminalKeyReader()

        if self.keyboard_gripper:
            self._initialize_manual_gripper_state()

    def request_stop(self, *_):
        self._stop = True

    # ------------------------------------------------------------------ #
    def _start_teach(self):
        """Put the LEADER into freedrive/teach mode for hand-guiding."""
        ok = self.ctrl_leader.teachMode()
        if ok is False:
            raise RuntimeError(
                "Leader teachMode() failed. Check the leader is in Remote Control "
                "mode and no protective stop is active."
            )
        self._teach_active = True
        print("  Leader is in TEACH MODE -- hand-guide it freely.")

    def _stop_teach(self):
        """End leader teach mode. Safe to call multiple times."""
        if self._teach_active:
            try:
                self.ctrl_leader.endTeachMode()
            finally:
                self._teach_active = False
            print("  Leader teach mode ended.")

    # ------------------------------------------------------------------ #
    def _initialize_manual_gripper_state(self):
        """Read the follower gripper once to establish its initial toggle state."""
        if self.foll_g is None:
            return
        try:
            position = self.foll_g.get_current_position()
            midpoint = (self.foll_g.get_open_position() + self.foll_g.get_closed_position()) / 2
            self._manual_gripper_closed = position >= midpoint
            self._manual_gripper_target = position
        except Exception as exc:
            # Default to open if the initial read is unavailable. The first 'g'
            # press will command a close operation.
            print(f"  [gripper] could not read initial follower position: {exc}")
            self._manual_gripper_closed = False
            self._manual_gripper_target = self.foll_g.get_open_position()
        self._write_gripper_status()

    def _write_gripper_status(self):
        """Publish the manual follower-gripper command for the data recorder.

        The write is atomic so collect_data.py never reads a partially written
        JSON file. This records command state, not a second gripper connection.
        """
        if not self.keyboard_gripper or self.gripper_status_file is None:
            return

        self.gripper_status_file.parent.mkdir(parents=True, exist_ok=True)
        status = {
            "commanded_closed": self._manual_gripper_closed,
            "target_position": self._manual_gripper_target,
            "updated_monotonic_ns": time.monotonic_ns(),
        }
        temporary_path = self.gripper_status_file.with_suffix(self.gripper_status_file.suffix + ".tmp")
        temporary_path.write_text(json.dumps(status), encoding="utf-8")
        os.replace(temporary_path, self.gripper_status_file)

    def _toggle_manual_gripper(self):
        """Toggle the follower gripper between calibrated open and closed positions."""
        if not self.keyboard_gripper or self.foll_g is None:
            return

        next_closed = not self._manual_gripper_closed
        target = self.foll_g.get_closed_position() if next_closed else self.foll_g.get_open_position()
        try:
            command_ok, commanded_position = self.foll_g.move(
                target, self.cfg.GRIP_SPEED, self.cfg.GRIP_FORCE
            )
        except Exception as exc:
            print(f"  [gripper] toggle failed: {exc}")
            return

        if not command_ok:
            print("  [gripper] toggle was not acknowledged; state was not changed.")
            return

        self._manual_gripper_closed = next_closed
        self._manual_gripper_target = commanded_position
        self._write_gripper_status()
        state_name = "CLOSED" if next_closed else "OPEN"
        print(f"  [gripper] commanded {state_name} (target position {commanded_position}).")

    def _handle_keyboard_input(self):
        """Handle non-blocking manual gripper key presses during mirroring."""
        if not self.keyboard_gripper:
            return
        for key in self._keyboard.read_keys():
            if key.lower() == "g":
                self._toggle_manual_gripper()

    # ------------------------------------------------------------------ #
    def _mirror_gripper_step(self, now):
        """Relay leader gripper position to follower, rate-limited + deadbanded."""
        if not self.mirror_gripper:
            return
        if now - self._last_grip_update < 1.0 / self.cfg.GRIP_UPDATE_HZ:
            return
        self._last_grip_update = now

        target = self.lead_g.get_current_position()   # 0..255
        if (self._last_grip_cmd is None or
                abs(target - self._last_grip_cmd) >= self.cfg.GRIP_DEADBAND):
            # non-blocking move so the arm loop is not stalled
            self.foll_g.move(target, self.cfg.GRIP_SPEED, self.cfg.GRIP_FORCE)
            self._last_grip_cmd = target

    # ------------------------------------------------------------------ #
    def run(self):
        cfg = self.cfg
        period = 1.0 / cfg.RATE_HZ

        print("\n" + "=" * 70)
        print("MIRROR RUNNING -- follower is tracking the leader.")
        print("Hand-guide the LEADER in freedrive. Ctrl-C to stop.")
        if self.keyboard_gripper:
            print("Press 'g' in this terminal to toggle the UR5e gripper open/closed.")
        print("=" * 70 + "\n")

        if self.keyboard_gripper:
            self._keyboard.start()

        self._start_teach()   # put leader in freedrive so it can be hand-guided

        # Capture a matched reference moment. The two robots may be in different
        # physical locations and slightly different poses; that is expected. From
        # this point onward, follower motion equals leader motion *relative to*
        # these startup TCP poses.
        print("  Capturing startup reference poses...")
        leader_start_pose = self.rtde_leader.getActualTCPPose()
        follower_start_pose = self.rtde_follower.getActualTCPPose()
        print("  Reference poses captured. Follower will mirror leader motion ")
        print("  relative to its own starting pose.")

        held_last = False
        n = 0
        try:
            while not self._stop:
                loop_start = time.monotonic()
                self._handle_keyboard_input()

                lead_pose = self.rtde_leader.getActualTCPPose()
                target = relative_pose_target(
                    leader_start_pose,
                    lead_pose,
                    follower_start_pose,
                    cfg.FRAME_TRANSFORM,
                )
                foll_pose = self.rtde_follower.getActualTCPPose()

                d = dist(target, foll_pose)
                if d <= cfg.MAX_CHASE_DIST_M:
                    self.ctrl_follower.servoL(
                        target, cfg.SERVO_SPEED, cfg.SERVO_ACC,
                        period, cfg.SERVO_LOOKAHEAD, cfg.SERVO_GAIN)
                    if held_last:
                        print("  [guard] target back in range -- resuming.")
                        held_last = False
                else:
                    # hold: stop servoing, wait for leader to come back in range
                    self.ctrl_follower.servoStop()
                    if not held_last:
                        print(f"  [guard] follower is {d:.2f} m from relative target (> "
                              f"{cfg.MAX_CHASE_DIST_M} m) -- HOLDING follower.")
                        held_last = True

                self._mirror_gripper_step(loop_start)

                n += 1
                sleep_t = period - (time.monotonic() - loop_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)
                # else: loop overran; RTDE servoL tolerates jitter, so just continue

        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    # ------------------------------------------------------------------ #
    def close(self):
        print("\nStopping mirror...")
        self._keyboard.close()
        # 1) END LEADER TEACH MODE FIRST -- leaving it on releases the leader's
        #    brakes. This must happen no matter how we exit.
        self._stop_teach()
        # 2) stop follower servo motion
        try:
            self.ctrl_follower.servoStop()
        except Exception:
            pass
        # 3) release both control scripts and disconnect both control interfaces
        for c, name in ((self.ctrl_leader, "leader control"),
                        (self.ctrl_follower, "follower control")):
            try:
                c.stopScript()
            except Exception:
                pass
            try:
                c.disconnect()
            except Exception:
                pass
        # 4) receive interfaces
        for r in (self.rtde_leader, self.rtde_follower):
            try:
                r.disconnect()
            except Exception:
                pass
        # 5) grippers
        for g, name in ((self.lead_g, "leader gripper"),
                        (self.foll_g, "follower gripper")):
            if g is not None:
                try:
                    g.disconnect()
                    print(f"  {name} disconnected.")
                except Exception:
                    pass
        print("Mirror stopped. Leader freedrive released. All interfaces closed.")


# =============================================================================
# main
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Leader/follower TCP-pose mirror.")
    p.add_argument("--no-gripper", action="store_true",
                   help="Arm pose only; do not connect either gripper.")
    p.add_argument(
        "--keyboard-gripper",
        action="store_true",
        help="Press 'g' to toggle the follower gripper instead of mirroring the leader gripper.",
    )
    p.add_argument(
        "--gripper-status-file",
        type=Path,
        default=Path("gripper_status.json"),
        help="JSON file used by collect_data.py to record keyboard gripper commands.",
    )
    args = p.parse_args()

    if args.no_gripper and args.keyboard_gripper:
        p.error("--no-gripper and --keyboard-gripper cannot be used together.")

    cfg = Config()
    if "IP_HERE" in cfg.LEADER_IP or "IP_HERE" in cfg.FOLLOWER_IP:
        print("ERROR: set LEADER_IP and FOLLOWER_IP in Config first.", file=sys.stderr)
        sys.exit(1)

    mirror = Mirror(
        cfg,
        mirror_gripper=not args.no_gripper and not args.keyboard_gripper,
        keyboard_gripper=args.keyboard_gripper,
        gripper_status_file=args.gripper_status_file if args.keyboard_gripper else None,
    )
    # allow clean Ctrl-C
    signal.signal(signal.SIGINT, mirror.request_stop)
    mirror.run()


if __name__ == "__main__":
    main()
