"""
Configuration for the robot controller.
Update it before using any other script.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Final
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# Module-level constants
DEFAULT_ROBOT_TYPE: Final[str] = "lekiwi" # "so100", "so101", "lekiwi"
DEFAULT_SERIAL_PORT: Final[str] = "/dev/tty.usbmodem58FD0168731" # only for SO ARM
DEFAULT_REMOTE_IP: Final[str] = "192.168.1.1" # only for LeKiwi

# Camera configuration constants
# Can also be different for different cameras, set it in lerobot_config
DEFAULT_CAMERA_FPS: Final[int] = 30
DEFAULT_CAMERA_WIDTH: Final[int] = 640
DEFAULT_CAMERA_HEIGHT: Final[int] = 360

@dataclass
class RobotConfig:
    """
    Configuration for the robot controller.
    
    This dataclass contains all configuration parameters needed for robot operation,
    including hardware settings, kinematic parameters, and movement constants.
    """

    # This will hold the configuration for the `lerobot` robot instance.
    # It's structured as a dictionary to be passed to `make_robot_from_config` or a similar factory.
    lerobot_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": DEFAULT_ROBOT_TYPE,
            "port": DEFAULT_SERIAL_PORT,
            "remote_ip": DEFAULT_REMOTE_IP,
            "cameras": {
                "front": OpenCVCameraConfig(
                    index_or_path=0,
                    fps=DEFAULT_CAMERA_FPS,
                    width=DEFAULT_CAMERA_WIDTH,
                    height=DEFAULT_CAMERA_HEIGHT,
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=1,
                    fps=DEFAULT_CAMERA_FPS,
                    width=DEFAULT_CAMERA_WIDTH,
                    height=DEFAULT_CAMERA_HEIGHT,
                ),
                "top": OpenCVCameraConfig(
                    index_or_path=2,
                    fps=DEFAULT_CAMERA_FPS,
                    width=DEFAULT_CAMERA_WIDTH,
                    height=DEFAULT_CAMERA_HEIGHT,
                ),
            },
        }
    )

    # Mapping from lerobot's normalized motor outputs (-100 to 100 or 0 to 100) to degrees.
    # Format: {motor_name: (norm_min, norm_max, deg_min, deg_max)}
    # Use check_positions.py, move your robot to 0, 90, 180 degree positions 
    # and insert the corresponding normalized values here
    # You can use any 2 points per motor to interpolate, but wider range is better
    # TODO: find a simpler way to calibrate the robot
    MOTOR_NORMALIZED_TO_DEGREE_MAPPING: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan":  (-91.7, 99.5, 0.0, 180.0),
            "shoulder_lift": (-89.4, 99.4, 0, 180.0),
            "elbow_flex":    (96.5, -92.7, 0, 180.0),
            "wrist_flex":    (-90.0, 90.0, -90.0, 90.0),
            "wrist_roll":    (100, -100, -90, 90),
            "gripper":       (31.0, 100.0, 0.0, 100.0),
        }
    )

    # Movement constants for smooth interpolation
    MOVEMENT_CONSTANTS: Dict[str, Any] = field(
        default_factory=lambda: {
            "DEGREES_PER_STEP": 1.5,           # Degrees per interpolation step
            "MAX_INTERPOLATION_STEPS": 150,    # Maximum number of interpolation steps
            "STEP_DELAY_SECONDS": 0.01,        # Delay between interpolation steps (100Hz)
        }
    )

    # Robot description for AI/LLM context
    robot_description: str = ("""
Follow these instructions precisely. Never deviate.

You control a 3D printed robot with 5 DOF + gripper. Max forward reach ~250 mm.
Shoulder and elbow links are 12 cm and 14 cm. Gripper fingers ~8 cm.
Use these to estimate distances. E.g., if the object is near but not in the gripper, you can safely move 5–10 cm forward.

Robot has 3 cameras:
- front: at the base, looks forward
- wrist: close view of gripper
- top view: shows whole robot

Instructions:
- Move slowly and iteratively
- Close gripper completely to grab objects
- Check results after each move before proceeding
- When the object inside your gripper it will not be visible on top and front cameras and will cover the whole view for the wrist one
- Split into smaller steps and reanalyze after each one
- Use only the latest images to evaluate success
- Always plan movements to avoid collisions
- Move above object with gripper tilted up (10–15°) to avoid collisions. Stay >25 cm above ground when moving or rotating
- Never move with gripper near the ground
- Drop and restart plan if unsure or failed
"""
    )

    # Kinematic parameters for different robot types
    # You generally don't need to change these unless you have a custom robot
    KINEMATIC_PARAMS: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "default": {
                "L1": 117.0,  # Shoulder to elbow length (mm)
                "L2": 136.0,  # Elbow to wrist length (mm)
                "BASE_HEIGHT_MM": 120.0,
                "SHOULDER_MOUNT_OFFSET_MM": 32.0,
                "ELBOW_MOUNT_OFFSET_MM": 4.0,
                "SPATIAL_LIMITS": {
                    "x": (-20.0, 250.0),  # Forward/backward limits
                    "z": (30.0, 370.0),   # Up/down limits
                }
            },
            "lekiwi": {
                "L1": 117.0,
                "L2": 136.0,
                "BASE_HEIGHT_MM": 210.0, # LeKiwi is 9cm elevated, adjust if using different wheels
                "SHOULDER_MOUNT_OFFSET_MM": 32.0,
                "ELBOW_MOUNT_OFFSET_MM": 4.0,
                "SPATIAL_LIMITS": {
                    "x": (-20.0, 250.0),
                    "z": (30.0, 370.0),
                }
            }
        }
    )

    # Predefined robot positions for quick access
    PRESET_POSITIONS: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "1": { "gripper": 0.0, "wrist_roll": 90.0, "wrist_flex": 0.0, "elbow_flex": 0.0, "shoulder_lift": 0.0, "shoulder_pan": 90.0 },
            "2": { "gripper": 0.0, "wrist_roll": 90.0, "wrist_flex": 0.0, "elbow_flex": 45.0, "shoulder_lift": 45.0, "shoulder_pan": 90.0 },
            "3": { "gripper": 40.0, "wrist_roll": 90.0, "wrist_flex": 90.0, "elbow_flex": 45.0, "shoulder_lift": 45.0, "shoulder_pan": 90.0 },
            "4": { "gripper": 40.0, "wrist_roll": 90.0, "wrist_flex": -60.0, "elbow_flex": 20.0, "shoulder_lift": 80.0, "shoulder_pan": 90.0 },
        }
    )

# Create a global instance
robot_config = RobotConfig()