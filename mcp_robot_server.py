"""
MCP server to control the robot.
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional, Union

import numpy as np
from PIL import Image as PILImage

from mcp.server.fastmcp import FastMCP, Image

from robot_controller import RobotController
from config import robot_config

import atexit
import traceback
import time


logging.basicConfig(level=logging.INFO, format="%(asctime)s MCP_Server %(levelname)s: %(message)s")
logger = logging.getLogger(__name__) # Use a named logger for MCP server specifics if any

# -----------------------------------------------------------------------------
# Initialise FastMCP server
# -----------------------------------------------------------------------------

mcp = FastMCP(
    name="SO-ARM100 robot controller",
    port = 3001 # can use any other port
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

_robot: Optional[RobotController] = None


def _np_to_mcp_image(arr_rgb: np.ndarray) -> Image:
    """Convert a numpy RGB image to MCP image format."""
    pil_img = PILImage.fromarray(arr_rgb)
    with io.BytesIO() as buf:
        pil_img.save(buf, format="JPEG")
        raw_data = buf.getvalue()
    return Image(data=raw_data, format="jpeg")


def get_robot() -> RobotController:
    """Lazy-initialise the global RobotController instance.

    We avoid creating the controller at import time so the MCP Inspector can
    start even if the hardware is not connected. The first tool/resource call
    that actually needs the robot will trigger the connection.
    """
    global _robot
    if _robot is None:   
        try:
            _robot = RobotController()
            logger.info(f"RobotController initialized.")

        except Exception as e:
            logger.error(f"MCP: FATAL - Error initializing robot: {e}", exc_info=True)
            raise SystemExit(f"MCP Server cannot start: RobotController failed to initialize ({e})")
            
    return _robot


def get_state_with_images(result_json: dict, is_movement: bool = False) -> List[Union[Image, dict, list]]:
    """Combine robot state with camera images into a unified response format.
    Returns a list containing:
    1. MCP images from all available cameras
    2. JSON with robot state and operation results

    Args:
        result_json: The operation result in JSON format
        is_movement: If True, adds a small delay before capturing images to ensure they're current
    """
    robot = get_robot()
    try:
        if is_movement:
            time.sleep(1.0)  # wait until the robot moved before capturing images
        
        raw_imgs = robot.get_camera_images()
        
        if not raw_imgs:
            logger.warning("MCP: No camera images returned from robot controller.")
            return [result_json, "Warning: No camera images available."]

        mcp_images = [_np_to_mcp_image(img) for img in raw_imgs.values()]
            
        # Keep only human_readable_state inside robot_state for clients
        result_json["robot_state"] = result_json["robot_state"]["human_readable_state"]

        # Return combined response
        # I am not sure if it is the official way to return both json and images in one response
        # But it works with most clients I tested
        return [result_json] + mcp_images
    except Exception as e:
        logger.error(f"Error getting camera images: {str(e)}")
        logger.error(traceback.format_exc())
        # If camera access fails, still return state with empty image list
        return [result_json] + ["Error getting camera images"]


# -----------------------------------------------------------------------------
# Tools – read-only
# -----------------------------------------------------------------------------

# Can be resource instead but some clients support only tools
# @mcp.resource("robot://description")
@mcp.tool(description="Get a description of the robot and instructions for the user. Run it before using any other tool.")
def get_initial_instructions() -> str:
    return robot_config.robot_description


@mcp.tool(description="Get current robot state with images from all cameras. Returns list of objects: json with results of the move and current state of the robot and images from all cameras")
def get_robot_state():
    robot = get_robot()
    move_result = robot.get_current_robot_state()
    result_json = move_result.to_json()
    logger.info(f"MCP: get_robot_state outcome: {result_json.get('status', 'success')}, Msg: {move_result.msg}")
    return get_state_with_images(result_json, is_movement=False)


# -----------------------------------------------------------------------------
# Tools – actuation
# -----------------------------------------------------------------------------

@mcp.tool(
        description="""
        Move the robot with intuitive controls.
        Args:
            move_gripper_up_mm (float, optional): Distance to move gripper up (positive) or down (negative) in mm
            move_gripper_forward_mm (float, optional): Distance to move gripper forward (positive) or backward (negative) in mm
            tilt_gripper_down_angle (float, optional): Angle to tilt gripper down (positive) or up (negative) in degrees
            rotate_gripper_counterclockwise_angle (float, optional): Angle to rotate gripper counterclockwise (positive) or clockwise (negative) in degrees
            rotate_robot_left_angle (float, optional): Angle to rotate entire robot counterclockwise/left (positive) or counterclockwise/right (negative) in degrees
        Expected input format:
        {
            "move_gripper_up_mm": "10", # Will move up 1 cm
            "move_gripper_forward_mm": "-5", # Will move backward 5 mm
            "tilt_gripper_down_angle": "10", # Will tilt gripper down 10 degrees
            "rotate_gripper_counterclockwise_angle": "-15", # Will rotate gripper clockwise 15 degrees
            "rotate_robot_left_angle": "15" # Will rotate robot counterclockwise (to the left) 15 degrees
        }
        Returns:
            list: List containing:
                - JSON object with:
                    - status: Optional status in case of error
                    - message: Optional message
                    - warnings: Optional list of any warnings
                    - robot_state: Current robot state in human readable format
                - Camera images
    """
        )
def move_robot(
    move_gripper_up_mm=None, 
    move_gripper_forward_mm=None, 
    tilt_gripper_down_angle=None, 
    rotate_gripper_counterclockwise_angle=None, 
    rotate_robot_left_angle=None
):
    robot = get_robot()
    logger.info(f"MCP Tool: move_robot received: up={move_gripper_up_mm}, fwd={move_gripper_forward_mm}, "
                f"tilt={tilt_gripper_down_angle}, grip_rot={rotate_gripper_counterclockwise_angle}, "
                f"robot_rot={rotate_robot_left_angle}")

    # All parameters are optional for execute_intuitive_move
    # Convert MCP tool parameters to match the arguments of execute_intuitive_move
    move_params = {
        "move_gripper_up_mm": float(move_gripper_up_mm) if move_gripper_up_mm is not None else None,
        "move_gripper_forward_mm": float(move_gripper_forward_mm) if move_gripper_forward_mm is not None else None,
        "tilt_gripper_down_angle": float(tilt_gripper_down_angle) if tilt_gripper_down_angle is not None else None,
        "rotate_gripper_counterclockwise_angle": float(rotate_gripper_counterclockwise_angle) if rotate_gripper_counterclockwise_angle is not None else None,
        "rotate_robot_left_angle": float(rotate_robot_left_angle) if rotate_robot_left_angle is not None else None,
    }
    
    # Filter out None values to pass only specified arguments to the robot controller method
    actual_move_params = {k: v for k, v in move_params.items() if v is not None}
    
    if not actual_move_params:
        current_state_result = robot.get_current_robot_state()
        result_json = current_state_result.to_json()
        result_json["message"] = "No movement parameters provided to move_robot tool."
        logger.info(f"MCP: move_robot outcome: {result_json.get('status', 'success')}, Msg: {result_json.get('message', '')}")
        return get_state_with_images(result_json, is_movement=False)

    move_execution_result = robot.execute_intuitive_move(**actual_move_params)
    result_json = move_execution_result.to_json()
    
    logger.info(f"MCP: move_robot final outcome: {result_json.get('status', 'success')}, Msg: {result_json.get('message', '')}, Warnings: {len(result_json.get('warnings', []))}")
    
    return get_state_with_images(result_json, is_movement=True)


@mcp.tool(description="Control the robot's gripper openness from 0% (completely closed) to 100% (completely open). Expected input format: {gripper_openness_pct: '50'}. Returns list of objects: json with results of the move and current state of the robot and images from all cameras")
def control_gripper(gripper_openness_pct):
    robot = get_robot()
    
    try:
        openness = float(gripper_openness_pct)
        logger.info(f"MCP Tool: control_gripper called with openness={gripper_openness_pct}%")
        
        move_result = robot.set_joints_absolute({'gripper': openness})
        result_json = move_result.to_json()
        logger.info(f"MCP: control_gripper outcome: {result_json.get('status', 'success')}, Msg: {move_result.msg}, Warnings: {len(move_result.warnings)}")
        return get_state_with_images(result_json, is_movement=True)
        
    except (ValueError, TypeError) as e:
        logger.error(f"MCP: control_gripper received invalid input: {gripper_openness_pct}, error: {str(e)}")
        return {"status": "error", "message": f"Invalid gripper openness value: {str(e)}"}


# -----------------------------------------------------------------------------
# Graceful shutdown
# -----------------------------------------------------------------------------

def _cleanup():
    """Disconnect from hardware on server shutdown."""
    global _robot
    if _robot is not None:
        try:
            _robot.disconnect()
        except Exception as e_disc:
            logger.error(f"MCP: Exception during _robot.disconnect(): {e_disc}", exc_info=True)

atexit.register(_cleanup)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting MCP Robot Server...")
    try:
        mcp.run()
    except SystemExit as e:
        logger.error(f"MCP Server failed to start: {e}")
    except Exception as e_main:
        logger.error(f"MCP Server CRITICAL RUNTIME ERROR: {e_main}", exc_info=True) 
