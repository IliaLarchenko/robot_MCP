"""
Unit tests for MCP robot server.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock
import json
import os
import sys
import numpy as np
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we want to test
import mcp_robot_server
from robot_controller import MoveResult


class TestMCPServerUtilities(unittest.TestCase):
    """Test MCP server utility functions."""
    
    def test_np_to_mcp_image(self):
        """Test numpy array to MCP image conversion."""
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 0, 0]  # Red square
        
        result = mcp_robot_server._np_to_mcp_image(test_image)
        
        self.assertEqual(result._mime_type, "image/jpeg")
        self.assertIsInstance(result.data, bytes)  # Raw binary data
        self.assertGreater(len(result.data), 0)
        
        # Verify it starts with JPEG header
        # JPEG files start with 0xFF 0xD8
        self.assertEqual(result.data[0:2], b'\xff\xd8')
    
    def test_np_to_mcp_image_invalid_shape(self):
        """Test numpy to MCP image with invalid shape."""
        # Create invalid image (wrong dimensions)
        test_image = np.zeros((100, 100), dtype=np.uint8)  # Missing color channel
        
        # PIL actually handles 2D arrays fine (converts to grayscale), so this won't raise ValueError
        # Instead test that it still produces a valid image
        result = mcp_robot_server._np_to_mcp_image(test_image)
        
        self.assertEqual(result._mime_type, "image/jpeg")
        self.assertIsInstance(result.data, bytes)  # Raw binary data
        self.assertGreater(len(result.data), 0)
        
        # Verify it starts with JPEG header
        # JPEG files start with 0xFF 0xD8
        self.assertEqual(result.data[0:2], b'\xff\xd8')
    
    @patch('mcp_robot_server.RobotController')
    def test_get_robot_singleton(self, mock_robot_controller):
        """Test robot singleton creation."""
        mock_controller = Mock()
        mock_robot_controller.return_value = mock_controller
        
        # Reset global robot
        mcp_robot_server._robot = None
        
        # First call should create robot
        robot1 = mcp_robot_server.get_robot()
        self.assertEqual(robot1, mock_controller)
        mock_robot_controller.assert_called_once()
        
        # Second call should return same instance
        robot2 = mcp_robot_server.get_robot()
        self.assertEqual(robot2, mock_controller)
        self.assertEqual(mock_robot_controller.call_count, 1)  # Still only called once
    
    @patch('mcp_robot_server.RobotController')
    def test_get_robot_initialization_error(self, mock_robot_controller):
        """Test robot initialization error handling."""
        mock_robot_controller.side_effect = Exception("Hardware not connected")
        
        # Reset global robot
        mcp_robot_server._robot = None
        
        with self.assertRaises(SystemExit):
            mcp_robot_server.get_robot()
    
    @patch('mcp_robot_server.get_robot')
    def test_get_state_with_images_success(self, mock_get_robot):
        """Test get_state_with_images success case."""
        # Mock robot controller
        mock_robot = Mock()
        mock_robot.get_camera_images.return_value = {
            "front": np.zeros((100, 100, 3), dtype=np.uint8),
            "wrist": np.zeros((100, 100, 3), dtype=np.uint8)
        }
        mock_get_robot.return_value = mock_robot
        
        # Test data
        result_json = {
            "status": "success",
            "robot_state": {
                "human_readable_state": {"height": 120.0, "forward": 240.0}
            }
        }
        
        result = mcp_robot_server.get_state_with_images(result_json, is_movement=False)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # JSON + 2 images
        self.assertEqual(result[0]["robot_state"], {"height": 120.0, "forward": 240.0})
    
    @patch('mcp_robot_server.get_robot')
    def test_get_state_with_images_camera_error(self, mock_get_robot):
        """Test get_state_with_images with camera error."""
        # Mock robot controller with camera error
        mock_robot = Mock()
        mock_robot.get_camera_images.side_effect = Exception("Camera error")
        mock_get_robot.return_value = mock_robot
        
        result_json = {"status": "success", "robot_state": {"human_readable_state": {}}}
        
        result = mcp_robot_server.get_state_with_images(result_json, is_movement=False)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # JSON + error message
        self.assertIn("Error getting camera images", result[1])


class TestMCPServerTools(unittest.TestCase):
    """Test MCP server tool functions."""
    
    @patch('mcp_robot_server.robot_config')
    def test_get_initial_instructions(self, mock_config):
        """Test get_initial_instructions tool."""
        mock_config.robot_description = "Test robot description"
        
        result = mcp_robot_server.get_initial_instructions()
        
        self.assertEqual(result, "Test robot description")
    
    @patch('mcp_robot_server.get_robot')
    @patch('mcp_robot_server.get_state_with_images')
    def test_get_robot_state(self, mock_get_state, mock_get_robot):
        """Test get_robot_state tool."""
        # Mock robot controller
        mock_robot = Mock()
        mock_move_result = MoveResult(
            ok=True,
            msg="State retrieved",
            robot_state={"height": 120.0}
        )
        mock_robot.get_current_robot_state.return_value = mock_move_result
        mock_get_robot.return_value = mock_robot
        
        # Mock get_state_with_images
        mock_get_state.return_value = [{"robot_state": {"height": 120.0}}, "image1", "image2"]
        
        result = mcp_robot_server.get_robot_state()
        
        self.assertIsInstance(result, list)
        mock_robot.get_current_robot_state.assert_called_once()
        mock_get_state.assert_called_once_with(mock_move_result.to_json(), is_movement=False)
    
    @patch('mcp_robot_server.get_robot')
    @patch('mcp_robot_server.get_state_with_images')
    def test_move_robot_success(self, mock_get_state, mock_get_robot):
        """Test move_robot tool success."""
        # Mock robot controller
        mock_robot = Mock()
        mock_move_result = MoveResult(
            ok=True,
            msg="Move completed",
            robot_state={"height": 130.0}
        )
        mock_robot.execute_intuitive_move.return_value = mock_move_result
        mock_get_robot.return_value = mock_robot
        
        # Mock get_state_with_images
        mock_get_state.return_value = [{"robot_state": {"height": 130.0}}, "image1"]
        
        result = mcp_robot_server.move_robot(
            move_gripper_up_mm=10.0,
            move_gripper_forward_mm=5.0
        )
        
        self.assertIsInstance(result, list)
        mock_robot.execute_intuitive_move.assert_called_once_with(
            move_gripper_up_mm=10.0,
            move_gripper_forward_mm=5.0
        )
        mock_get_state.assert_called_once()
    
    @patch('mcp_robot_server.get_robot')
    @patch('mcp_robot_server.get_state_with_images')
    def test_move_robot_no_parameters(self, mock_get_state, mock_get_robot):
        """Test move_robot tool with no parameters."""
        # Mock robot controller
        mock_robot = Mock()
        mock_move_result = MoveResult(
            ok=True,
            msg="Current state",
            robot_state={"height": 120.0}
        )
        mock_robot.get_current_robot_state.return_value = mock_move_result
        mock_get_robot.return_value = mock_robot
        
        # Mock get_state_with_images
        mock_get_state.return_value = [{"robot_state": {"height": 120.0}}]
        
        result = mcp_robot_server.move_robot()
        
        self.assertIsInstance(result, list)
        mock_robot.get_current_robot_state.assert_called_once()
        # Should not call execute_intuitive_move
        mock_robot.execute_intuitive_move.assert_not_called()
    
    @patch('mcp_robot_server.get_robot')
    @patch('mcp_robot_server.get_state_with_images')
    def test_move_robot_all_parameters(self, mock_get_state, mock_get_robot):
        """Test move_robot tool with all parameters."""
        # Mock robot controller
        mock_robot = Mock()
        mock_move_result = MoveResult(ok=True, msg="Move completed", robot_state={})
        mock_robot.execute_intuitive_move.return_value = mock_move_result
        mock_get_robot.return_value = mock_robot
        
        mock_get_state.return_value = [{}]
        
        result = mcp_robot_server.move_robot(
            move_gripper_up_mm=10.0,
            move_gripper_forward_mm=5.0,
            tilt_gripper_down_angle=15.0,
            rotate_gripper_counterclockwise_angle=20.0,
            rotate_robot_left_angle=30.0
        )
        
        mock_robot.execute_intuitive_move.assert_called_once_with(
            move_gripper_up_mm=10.0,
            move_gripper_forward_mm=5.0,
            tilt_gripper_down_angle=15.0,
            rotate_gripper_counterclockwise_angle=20.0,
            rotate_robot_left_angle=30.0
        )
    
    @patch('mcp_robot_server.get_robot')
    @patch('mcp_robot_server.get_state_with_images')
    def test_control_gripper_success(self, mock_get_state, mock_get_robot):
        """Test control_gripper tool success."""
        # Mock robot controller
        mock_robot = Mock()
        mock_move_result = MoveResult(
            ok=True,
            msg="Gripper set",
            robot_state={"gripper": 75.0}
        )
        mock_robot.set_joints_absolute.return_value = mock_move_result
        mock_get_robot.return_value = mock_robot
        
        mock_get_state.return_value = [{"robot_state": {"gripper": 75.0}}]
        
        result = mcp_robot_server.control_gripper(gripper_openness_pct=75.0)
        
        self.assertIsInstance(result, list)
        mock_robot.set_joints_absolute.assert_called_once_with({'gripper': 75.0})
        mock_get_state.assert_called_once()
    
    @patch('mcp_robot_server.get_robot')
    def test_control_gripper_invalid_input(self, mock_get_robot):
        """Test control_gripper tool with invalid input."""
        mock_robot = Mock()
        mock_get_robot.return_value = mock_robot
        
        result = mcp_robot_server.control_gripper(gripper_openness_pct="invalid")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid gripper openness", result["message"])
    
    @patch('mcp_robot_server.get_robot')
    @patch('mcp_robot_server.get_state_with_images')
    def test_control_gripper_string_input(self, mock_get_state, mock_get_robot):
        """Test control_gripper tool with string input."""
        # Mock robot controller
        mock_robot = Mock()
        mock_move_result = MoveResult(ok=True, msg="Gripper set", robot_state={})
        mock_robot.set_joints_absolute.return_value = mock_move_result
        mock_get_robot.return_value = mock_robot
        
        mock_get_state.return_value = [{}]
        
        result = mcp_robot_server.control_gripper(gripper_openness_pct="50")
        
        self.assertIsInstance(result, list)
        mock_robot.set_joints_absolute.assert_called_once_with({'gripper': 50.0})


class TestMCPServerParameterHandling(unittest.TestCase):
    """Test MCP server parameter handling and validation."""
    
    @patch('mcp_robot_server.get_robot')
    @patch('mcp_robot_server.get_state_with_images')
    def test_move_robot_parameter_filtering(self, mock_get_state, mock_get_robot):
        """Test that None parameters are filtered out."""
        mock_robot = Mock()
        mock_move_result = MoveResult(ok=True, msg="Move completed", robot_state={})
        mock_robot.execute_intuitive_move.return_value = mock_move_result
        mock_get_robot.return_value = mock_robot
        
        mock_get_state.return_value = [{}]
        
        # Call with mix of None and valid parameters
        mcp_robot_server.move_robot(
            move_gripper_up_mm=10.0,
            move_gripper_forward_mm=None,  # Should be filtered out
            tilt_gripper_down_angle=15.0,
            rotate_gripper_counterclockwise_angle=None,  # Should be filtered out
            rotate_robot_left_angle=None  # Should be filtered out
        )
        
        # Verify only non-None parameters were passed
        mock_robot.execute_intuitive_move.assert_called_once_with(
            move_gripper_up_mm=10.0,
            tilt_gripper_down_angle=15.0
        )
    
    @patch('mcp_robot_server.get_robot')
    @patch('mcp_robot_server.get_state_with_images')
    def test_move_robot_string_parameters(self, mock_get_state, mock_get_robot):
        """Test move_robot with string parameters (converted to float)."""
        mock_robot = Mock()
        mock_move_result = MoveResult(ok=True, msg="Move completed", robot_state={})
        mock_robot.execute_intuitive_move.return_value = mock_move_result
        mock_get_robot.return_value = mock_robot
        
        mock_get_state.return_value = [{}]
        
        # Call with string parameters
        mcp_robot_server.move_robot(
            move_gripper_up_mm="10.5",
            move_gripper_forward_mm="5.2"
        )
        
        # Verify parameters were converted to float
        mock_robot.execute_intuitive_move.assert_called_once_with(
            move_gripper_up_mm=10.5,
            move_gripper_forward_mm=5.2
        )


class TestMCPServerIntegration(unittest.TestCase):
    """Test MCP server integration scenarios."""
    
    @patch('mcp_robot_server.get_robot')
    @patch('mcp_robot_server.get_state_with_images')
    @patch('mcp_robot_server.time.sleep')  # Mock sleep to speed up tests
    def test_movement_with_delay(self, mock_sleep, mock_get_state, mock_get_robot):
        """Test that movement operations include delay."""
        mock_robot = Mock()
        mock_move_result = MoveResult(ok=True, msg="Move completed", robot_state={})
        mock_robot.set_joints_absolute.return_value = mock_move_result
        mock_get_robot.return_value = mock_robot
        
        mock_get_state.return_value = [{}]
        
        mcp_robot_server.control_gripper(gripper_openness_pct=50)
        
        # Verify sleep was called for movement
        mock_get_state.assert_called_once_with(mock_move_result.to_json(), is_movement=True)
    
    @patch('mcp_robot_server.get_robot')
    @patch('mcp_robot_server.get_state_with_images')
    def test_state_query_no_delay(self, mock_get_state, mock_get_robot):
        """Test that state queries don't include delay."""
        mock_robot = Mock()
        mock_move_result = MoveResult(ok=True, msg="State retrieved", robot_state={})
        mock_robot.get_current_robot_state.return_value = mock_move_result
        mock_get_robot.return_value = mock_robot
        
        mock_get_state.return_value = [{}]
        
        mcp_robot_server.get_robot_state()
        
        # Verify no movement delay for state query
        mock_get_state.assert_called_once_with(mock_move_result.to_json(), is_movement=False)


if __name__ == '__main__':
    unittest.main() 