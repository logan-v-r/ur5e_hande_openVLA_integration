#!/usr/bin/env python3
"""
collect_openvla_realsense_simple.py
===================================
Read-only 5 Hz UR5e + RealSense recorder for OpenVLA dataset collection.

Run this in a SECOND terminal while your 125 Hz mirror script controls the UR5e.
This script never creates a control or I/O interface and never commands the robot
or gripper to avoid contention issues.

Output is intentionally RAW, not RLDS. A later cleanup/conversion script (look at data_processing/) can:
  * drop stale/missing camera records
  * remove near-zero actions
  * reject unsafe or incomplete recordings
  * review task success
  * convert retained steps to the OpenVLA/RLDS image + 8D-action format

Folder layout
-------------
episode_.../
  episode_metadata.json   # one-time task, camera, and action-convention details
  steps.jsonl             # one compact JSON record per sampled timestep
  images/000000.jpg       # RGB observations
  COMPLETE.json           # written only after a clean shutdown

Dependencies
------------
pip install ur-rtde numpy scipy opencv-python pyrealsense2

Example
-------
python collect_data_gripper.py \\
  --follower-ip 192.168.1.102 \\
  --output-root ~/datasets/ur5e_openvla_raw \\
  --instruction "Pick up the red block and place it in the blue bin." \\
  --task-id pick_red_block_blue_bin \\
  --tcp-description "UR5e + Hand-E calibrated TCP"

Optional gripper observation (read-only):
  --gripper-status-file ./gripper_status.json
  --gripper-command-output-bit 0 --gripper-state-input-bit 0

When a mirror gripper status file is provided, it is used before I/O bits.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation as R

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


# -----------------------------------------------------------------------------
# Small utility functions
# -----------------------------------------------------------------------------
def utc_now() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """Convert common recorder values into types accepted by json.dumps()."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def try_read(read_function: Any) -> Any:
    """Run one non-critical RTDE read and return None if it fails.

    A missing telemetry field should flag the current step for cleanup, not
    terminate an otherwise useful recording session.
    """
    try:
        return read_function()
    except Exception:
        return None


def get_bit(bitfield: Optional[int], bit_index: Optional[int]) -> Optional[int]:
    """Return one digital-I/O bit, or None when the source or bit is unavailable."""
    if bitfield is None or bit_index is None:
        return None
    return (int(bitfield) >> int(bit_index)) & 1


def read_gripper_status_file(status_file: Optional[Path]) -> Optional[dict[str, Any]]:
    """Read the most recent keyboard-gripper command written by mirror_relative_keyboard.py.

    The mirror writes this small JSON file atomically. Reading it avoids opening
    a second socket connection to the Robotiq gripper from the data recorder.
    """
    if status_file is None:
        return None

    try:
        payload = json.loads(status_file.expanduser().read_text(encoding="utf-8"))
        commanded_closed = payload.get("commanded_closed")
        if not isinstance(commanded_closed, bool):
            return None
        return {
            "closed": float(commanded_closed),
            "target_position": payload.get("target_position"),
        }
    except Exception:
        return None


def tcp_delta(pose_a: list[float], pose_b: list[float]) -> list[float]:
    """Return [dx, dy, dz, drx, dry, drz] from actual follower poses."""
    a = np.asarray(pose_a, dtype=np.float64)
    b = np.asarray(pose_b, dtype=np.float64)
    translation = b[:3] - a[:3]
    rotation = (R.from_rotvec(b[3:6]) * R.from_rotvec(a[3:6]).inv()).as_rotvec()
    return np.concatenate((translation, rotation)).tolist()


def intrinsics_dict(intrinsics: Any) -> dict[str, Any]:
    """Keep only the camera intrinsic values useful for later review."""
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
    }


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
@dataclass
class Config:
    """Configuration values for one read-only recording session. Ensure these values match YOUR specific setup."""
    follower_ip: str
    output_root: Path
    instruction: str
    task_id: str
    tcp_description: str

    sample_hz: float = 5.0
    realsense_serial: Optional[str] = None
    color_width: int = 640
    color_height: int = 480
    color_fps: int = 30
    max_camera_age_ms: float = 100.0
    jpeg_quality: int = 95
    camera_warmup_s: float = 3.0

    # Optional only: the logger reads these values but never writes any I/O.
    gripper_command_output_bit: Optional[int] = None
    gripper_state_input_bit: Optional[int] = None
    gripper_command_close_level: int = 1
    gripper_state_closed_level: int = 1
    gripper_status_file: Optional[Path] = None



# -----------------------------------------------------------------------------
# RealSense capture thread
# -----------------------------------------------------------------------------
class RealSenseCamera:
    """Capture RealSense frames in a background thread for non-blocking recording.

    The recorder samples the newest available frame at its lower dataset rate.
    """

    def __init__(self, cfg: Config):
        """Configure the RealSense pipeline and initialize thread-safe frame storage."""
        if rs is None:
            raise RuntimeError("pyrealsense2 is required. Install it in this environment first.")

        self.cfg = cfg
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        if cfg.realsense_serial:
            self.rs_config.enable_device(cfg.realsense_serial)

        self.rs_config.enable_stream(
            rs.stream.color, cfg.color_width, cfg.color_height, rs.format.rgb8, cfg.color_fps
        )
        self.profile: Any = None
        self.metadata: dict[str, Any] = {}
        self._latest_frame: Optional[dict[str, Any]] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._capture_frames, daemon=True)

    def start(self) -> None:
        """Start the RealSense pipeline and begin background frame capture."""
        self.profile = self.pipeline.start(self.rs_config)
        device = self.profile.get_device()
        self.metadata["serial"] = device.get_info(rs.camera_info.serial_number)
        self.metadata["name"] = device.get_info(rs.camera_info.name)

        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.metadata["color_intrinsics"] = intrinsics_dict(color_profile.get_intrinsics())
        self.metadata["color_stream"] = {
            "width": self.cfg.color_width,
            "height": self.cfg.color_height,
            "fps": self.cfg.color_fps,
        }

        self._thread.start()

    def _capture_frames(self) -> None:
        """Read RealSense frames continuously and retain only the newest one."""
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                received_ns = time.monotonic_ns()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # OpenCV writes BGR JPEGs. The later converter should read and convert BGR -> RGB.
                color_rgb = np.asanyarray(color_frame.get_data()).copy()
                color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                packet = {
                    "color_bgr": color_bgr,
                    "received_monotonic_ns": received_ns,
                    "device_timestamp_ms": float(color_frame.get_timestamp()),
                    "frame_number": int(color_frame.get_frame_number()),
                }
                with self._lock:
                    self._latest_frame = packet
            except Exception:
                # Keep the recorder alive. Quality flags will expose stale/missing frames.
                time.sleep(0.02)

    def get_latest_frame(self) -> Optional[dict[str, Any]]:
        """Return a safe copy of the newest camera frame, or None if unavailable."""
        with self._lock:
            if self._latest_frame is None:
                return None
            packet = dict(self._latest_frame)
            packet["color_bgr"] = self._latest_frame["color_bgr"].copy()
            return packet

    def close(self) -> None:
        """Stop frame capture and release the RealSense pipeline."""
        self._stop.set()
        self._thread.join(timeout=1.5)
        try:
            self.pipeline.stop()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Episode recorder
# -----------------------------------------------------------------------------
class Recorder:
    """Record one UR5e demonstration as images plus synchronized raw telemetry.

    The recorder reads robot and camera data only. It never sends robot or
    gripper commands, so it can run beside the mirror-control process.
    """

    def __init__(self, cfg: Config):
        """Store configuration and initialize per-episode bookkeeping."""
        self.cfg = cfg
        self.stop_requested = False
        self.robot: Optional[RTDEReceiveInterface] = None
        self.camera: Optional[RealSenseCamera] = None
        self.episode_dir: Optional[Path] = None
        self.images_dir: Optional[Path] = None
        self.steps_file: Any = None
        self.pending_step: Optional[dict[str, Any]] = None
        self.step_index = 0
        self.image_index = 0
        self.start_ns: Optional[int] = None
        self.episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def request_stop(self, *_: Any) -> None:
        """Request a clean stop after the current sampling iteration."""
        self.stop_requested = True

    def _warm_up_camera(self) -> None:
        """Discard early RealSense frames before creating an episode.

        RealSense color output can briefly show incorrect color or unstable
        auto-exposure/white-balance immediately after the stream starts.
        This pause is not saved as part of the episode.
        """
        assert self.camera is not None
        warmup_s = self.cfg.camera_warmup_s
        if warmup_s <= 0:
            return

        print(f"Warming up RealSense for {warmup_s:.1f} seconds; early frames will not be saved.")
        deadline = time.monotonic() + warmup_s
        while not self.stop_requested and time.monotonic() < deadline:
            time.sleep(0.05)

        # Confirm that the camera has produced at least one frame before recording.
        if not self.stop_requested and self.camera.get_latest_frame() is None:
            raise RuntimeError("RealSense produced no color frame during warm-up.")

    def _create_episode(self) -> None:
        """Create an episode folder and save one-time metadata."""
        self.episode_dir = self.cfg.output_root.expanduser().resolve() / self.episode_id
        self.images_dir = self.episode_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=False)
        self.steps_file = (self.episode_dir / "steps.jsonl").open("w", encoding="utf-8", buffering=1)

        assert self.camera is not None
        metadata = {
            "schema_version": "ur5e_realsense_raw_v4_rgb_only",
            "episode_id": self.episode_id,
            "created_utc": utc_now(),
            "task_id": self.cfg.task_id,
            "language_instruction": self.cfg.instruction,
            "sample_hz": self.cfg.sample_hz,
            "action_definition": "actual TCP delta [dx,dy,dz,drx,dry,drz] + binary gripper delta + terminate",
            "tcp_description": self.cfg.tcp_description,
            "camera": self.camera.metadata,
            "gripper_io": {
                "command_output_bit": self.cfg.gripper_command_output_bit,
                "state_input_bit": self.cfg.gripper_state_input_bit,
                "command_close_level": self.cfg.gripper_command_close_level,
                "state_closed_level": self.cfg.gripper_state_closed_level,
                "status_file": None if self.cfg.gripper_status_file is None else str(self.cfg.gripper_status_file),
            },
        }
        (self.episode_dir / "episode_metadata.json").write_text(
            json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8"
        )

    def _save_image(self, image_bgr: np.ndarray) -> tuple[Optional[str], bool]:
        """Write one color image and return its relative path and save status."""
        assert self.episode_dir is not None
        relative_path = f"images/{self.image_index:06d}.jpg"
        ok = cv2.imwrite(
            str(self.episode_dir / relative_path),
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality],
        )
        if ok:
            self.image_index += 1
            return relative_path, True
        return None, False

    def _read_follower_state(self) -> dict[str, Any]:
        """Read the UR5e telemetry needed for actions, safety checks, and gripper I/O."""
        assert self.robot is not None

        # Keep the RTDE method names explicit so the saved fields are easy to review.
        return {
            "tcp_pose_actual": try_read(self.robot.getActualTCPPose),
            "joint_positions_actual": try_read(self.robot.getActualQ),
            "tcp_pose_target": try_read(self.robot.getTargetTCPPose),
            "robot_mode": try_read(self.robot.getRobotMode),
            "safety_mode": try_read(self.robot.getSafetyMode),
            "digital_input_bits": try_read(self.robot.getActualDigitalInputBits),
            "digital_output_bits": try_read(self.robot.getActualDigitalOutputBits),
        }

    def _read_gripper_state(self, robot_state: dict[str, Any]) -> dict[str, Any]:
        """Read the gripper command state from the mirror file or optional I/O bits.

        The keyboard-gripper status file is preferred because it reflects the
        same Robotiq command source used by mirror_relative_keyboard.py. I/O remains a
        fallback for installations that expose gripper state through UR bits.
        """
        mirror_status = read_gripper_status_file(self.cfg.gripper_status_file)
        if mirror_status is not None:
            return {
                "command_bit": None,
                "feedback_bit": None,
                "closed": mirror_status["closed"],
                "source": "mirror_keyboard_command",
                "target_position": mirror_status["target_position"],
            }

        command = get_bit(robot_state["digital_output_bits"], self.cfg.gripper_command_output_bit)
        feedback = get_bit(robot_state["digital_input_bits"], self.cfg.gripper_state_input_bit)

        if feedback is not None:
            closed = float(feedback == self.cfg.gripper_state_closed_level)
            source = "input_feedback"
        elif command is not None:
            closed = float(command == self.cfg.gripper_command_close_level)
            source = "output_command"
        else:
            closed = None
            source = None

        return {
            "command_bit": command,
            "feedback_bit": feedback,
            "closed": closed,
            "source": source,
            "target_position": None,
        }

    def _capture_step(self) -> dict[str, Any]:
        """Capture one camera observation and matching follower telemetry record."""
        assert self.camera is not None
        now_ns = time.monotonic_ns()
        frame = self.camera.get_latest_frame()

        image_path = None
        image_saved = False
        camera_age_ms = None
        camera_frame_number = None
        camera_timestamp_ms = None

        if frame is not None:
            camera_age_ms = (now_ns - frame["received_monotonic_ns"]) / 1e6
            image_path, image_saved = self._save_image(frame["color_bgr"])
            camera_frame_number = frame["frame_number"]
            camera_timestamp_ms = frame["device_timestamp_ms"]

        robot_state = self._read_follower_state()
        gripper = self._read_gripper_state(robot_state)
        safety_ok = robot_state["safety_mode"] in (None, 1)
        camera_ok = (
            frame is not None
            and image_saved
            and camera_age_ms is not None
            and camera_age_ms <= self.cfg.max_camera_age_ms
        )

        return {
            "step_index": self.step_index,
            "host_monotonic_ns": now_ns,
            "elapsed_s": None if self.start_ns is None else (now_ns - self.start_ns) / 1e9,
            "image": {
                "path": image_path,
                "saved": image_saved,
                "frame_number": camera_frame_number,
                "device_timestamp_ms": camera_timestamp_ms,
                "age_ms": camera_age_ms,
            },
            "follower": robot_state,
            "gripper": gripper,
            "quality": {
                "camera_ok": camera_ok,
                "tcp_available": robot_state["tcp_pose_actual"] is not None,
                "safety_ok": safety_ok,
            },
            "action_to_next_raw": None,
            "is_first": self.step_index == 0,
            "is_last": False,
            "is_terminal": False,
        }

    def _attach_action(self, current: dict[str, Any], following: dict[str, Any]) -> None:
        """Derive the current step action from actual motion to the next step."""
        pose_a = current["follower"]["tcp_pose_actual"]
        pose_b = following["follower"]["tcp_pose_actual"]
        gripper_a = current["gripper"]["closed"]
        gripper_b = following["gripper"]["closed"]

        if pose_a is None or pose_b is None:
            current["quality"]["action_valid"] = False
            current["quality"]["action_invalid_reason"] = "missing_tcp_pose"
            return

        delta = tcp_delta(pose_a, pose_b)
        gripper_delta = None if gripper_a is None or gripper_b is None else float(gripper_b - gripper_a)
        translation_m = float(np.linalg.norm(np.asarray(delta[:3])))
        rotation_rad = float(np.linalg.norm(np.asarray(delta[3:6])))
        small_motion = (
            translation_m < 0.002
            and rotation_rad < np.deg2rad(1.0)
            and (gripper_delta is None or abs(gripper_delta) < 0.5)
        )

        current["action_to_next_raw"] = {
            "tcp_delta_base_frame": delta,
            "gripper_delta_binary": gripper_delta,
            "terminate": 0.0,
            "action_8d_raw": None if gripper_delta is None else delta + [gripper_delta, 0.0],
            "dt_s": (following["host_monotonic_ns"] - current["host_monotonic_ns"]) / 1e9,
        }
        current["quality"].update(
            {
                "action_valid": True,
                "translation_delta_m": translation_m,
                "rotation_delta_rad": rotation_rad,
                "small_motion": small_motion,
            }
        )

    def _write_step(self, step: dict[str, Any]) -> None:
        """Append one completed raw step to the episode JSONL file."""
        assert self.steps_file is not None
        self.steps_file.write(json.dumps(to_jsonable(step), allow_nan=False) + "\n")
        self.steps_file.flush()

    def _write_final_step(self) -> None:
        """Write the final observation, which has no following action label."""
        if self.pending_step is None:
            return
        self.pending_step["is_last"] = True
        self.pending_step["is_terminal"] = True
        self.pending_step["quality"]["action_valid"] = False
        self.pending_step["quality"]["action_invalid_reason"] = "final_step_has_no_next_state"
        self._write_step(self.pending_step)
        self.pending_step = None

    def run(self) -> None:
        """Connect read-only interfaces and sample until the user stops recording."""
        self.robot = RTDEReceiveInterface(self.cfg.follower_ip)
        self.camera = RealSenseCamera(self.cfg)
        self.camera.start()
        self._warm_up_camera()
        if self.stop_requested:
            return

        # Create the episode only after warm-up so no unstable frames are saved.
        self._create_episode()
        self.start_ns = time.monotonic_ns()

        print(f"Recording at {self.cfg.sample_hz:.1f} Hz (read-only)")
        print(f"Episode: {self.episode_dir}")
        print("Press Ctrl-C once when the demonstration is complete.")

        period = 1.0 / self.cfg.sample_hz
        deadline = time.monotonic()
        try:
            while not self.stop_requested:
                now = time.monotonic()
                if now < deadline:
                    time.sleep(deadline - now)
                    continue
                deadline += period
                if now - deadline > period:
                    deadline = now + period

                step = self._capture_step()
                if self.pending_step is not None:
                    self._attach_action(self.pending_step, step)
                    self._write_step(self.pending_step)
                self.pending_step = step
                self.step_index += 1
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        """Flush the final step, mark the episode complete, and release resources."""
        try:
            self._write_final_step()
            if self.episode_dir is not None:
                complete = {
                    "recording_complete": True,
                    "closed_utc": utc_now(),
                    "steps_written": self.step_index,
                    "images_written": self.image_index,
                    "episode_status": "unreviewed",
                }
                (self.episode_dir / "COMPLETE.json").write_text(
                    json.dumps(complete, indent=2), encoding="utf-8"
                )
        finally:
            if self.steps_file is not None:
                self.steps_file.close()
            if self.camera is not None:
                self.camera.close()
            if self.robot is not None:
                try:
                    self.robot.disconnect()
                except Exception:
                    pass
            if self.episode_dir is not None:
                print(f"Saved episode: {self.episode_dir}")


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
def parse_args() -> Config:
    """Read command-line options and return validated recorder configuration."""
    parser = argparse.ArgumentParser(description="Read-only UR5e + RealSense raw data recorder.")
    parser.add_argument("--follower-ip", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--tcp-description", required=True)
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--realsense-serial", default=None)
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--color-fps", type=int, default=30)
    parser.add_argument("--max-camera-age-ms", type=float, default=100.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--camera-warmup-s",
        type=float,
        default=3.0,
        help="Seconds to discard RealSense frames before the episode begins (default: 3.0).",
    )
    parser.add_argument(
        "--gripper-status-file",
        type=Path,
        default=None,
        help="JSON state file written by mirror_relative.py in --keyboard-gripper mode.",
    )
    parser.add_argument("--gripper-command-output-bit", type=int, default=None)
    parser.add_argument("--gripper-state-input-bit", type=int, default=None)
    parser.add_argument("--gripper-command-close-level", type=int, choices=(0, 1), default=1)
    parser.add_argument("--gripper-state-closed-level", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    if not 1.0 <= args.sample_hz <= 30.0:
        parser.error("--sample-hz must be between 1 and 30.")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100.")
    if not 0.0 <= args.camera_warmup_s <= 30.0:
        parser.error("--camera-warmup-s must be between 0 and 30 seconds.")
    for bit, argument in (
        (args.gripper_command_output_bit, "--gripper-command-output-bit"),
        (args.gripper_state_input_bit, "--gripper-state-input-bit"),
    ):
        if bit is not None and not 0 <= bit <= 63:
            parser.error(f"{argument} must be between 0 and 63.")

    return Config(
        follower_ip=args.follower_ip,
        output_root=args.output_root,
        instruction=args.instruction,
        task_id=args.task_id,
        tcp_description=args.tcp_description,
        sample_hz=args.sample_hz,
        realsense_serial=args.realsense_serial,
        color_width=args.color_width,
        color_height=args.color_height,
        color_fps=args.color_fps,
        max_camera_age_ms=args.max_camera_age_ms,
        jpeg_quality=args.jpeg_quality,
        camera_warmup_s=args.camera_warmup_s,
        gripper_command_output_bit=args.gripper_command_output_bit,
        gripper_state_input_bit=args.gripper_state_input_bit,
        gripper_command_close_level=args.gripper_command_close_level,
        gripper_state_closed_level=args.gripper_state_closed_level,
        gripper_status_file=args.gripper_status_file,
    )


def main() -> None:
    """Create and run the recorder from command-line settings."""
    recorder = Recorder(parse_args())
    signal.signal(signal.SIGINT, recorder.request_stop)
    signal.signal(signal.SIGTERM, recorder.request_stop)
    recorder.run()


if __name__ == "__main__":
    main()
