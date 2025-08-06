"""
Unit tests for LLM providers.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json
import os
import sys
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_providers.base_provider import LLMProvider, LLMResponse
from llm_providers.factory import create_llm_provider
from llm_providers.claude_provider import ClaudeProvider
from llm_providers.gemini_provider import GeminiProvider


class TestLLMResponse(unittest.TestCase):
    """Test LLMResponse dataclass."""
    
    def test_default_initialization(self):
        """Test LLMResponse with default values."""
        response = LLMResponse()
        self.assertIsNone(response.content)
        self.assertIsNone(response.thinking)
        self.assertIsNone(response.tool_calls)
        self.assertEqual(response.provider, "")
        self.assertEqual(response.usage, {})
    
    def test_full_initialization(self):
        """Test LLMResponse with all values."""
        response = LLMResponse(
            content="Test content",
            thinking="Test thinking",
            tool_calls=[{"id": "1", "name": "test"}],
            provider="test_provider",
            usage={"tokens": 100}
        )
        self.assertEqual(response.content, "Test content")
        self.assertEqual(response.thinking, "Test thinking")
        self.assertEqual(response.tool_calls, [{"id": "1", "name": "test"}])
        self.assertEqual(response.provider, "test_provider")
        self.assertEqual(response.usage, {"tokens": 100})


class TestBaseProvider(unittest.TestCase):
    """Test base LLM provider functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation for testing
        class TestProvider(LLMProvider):
            @property
            def provider_name(self) -> str:
                return "Test"
            
            @property
            def supports_thinking(self) -> bool:
                return True
            
            def format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return tools
            
            def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return messages
            
            async def _generate_response_impl(self, messages: List[Dict[str, Any]], **kwargs) -> LLMResponse:
                return LLMResponse(content="test response", provider="Test")
        
        self.provider = TestProvider("test_key", "test_model")
    
    def test_initialization(self):
        """Test provider initialization."""
        self.assertEqual(self.provider.api_key, "test_key")
        self.assertEqual(self.provider.model, "test_model")
        self.assertEqual(self.provider.provider_name, "Test")
        self.assertTrue(self.provider.supports_thinking)
    
    def test_format_tools_for_llm(self):
        """Test format_tools_for_llm method."""
        tools = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "inputSchema": {"type": "object", "properties": {"param": {"type": "string"}}}
            }
        ]
        
        result = self.provider.format_tools_for_llm(tools)
        expected = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}}
            }
        ]
        self.assertEqual(result, expected)
    
    def test_format_tools_for_llm_missing_schema(self):
        """Test format_tools_for_llm with missing inputSchema."""
        tools = [{"name": "test_tool", "description": "Test tool"}]
        
        result = self.provider.format_tools_for_llm(tools)
        expected = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {"type": "object", "properties": {}}
            }
        ]
        self.assertEqual(result, expected)
    
    def test_format_tool_calls_for_execution(self):
        """Test format_tool_calls_for_execution method."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"param": "value"}'
                }
            }
        ]
        
        result = self.provider.format_tool_calls_for_execution(tool_calls)
        expected = [
            {
                "id": "call_1",
                "name": "test_tool",
                "input": {"param": "value"}
            }
        ]
        self.assertEqual(result, expected)
    
    def test_format_tool_calls_for_execution_invalid_json(self):
        """Test format_tool_calls_for_execution with invalid JSON."""
        tool_calls = [
            {
                "id": "call_1",
                "function": {
                    "name": "test_tool",
                    "arguments": 'invalid json'
                }
            }
        ]
        
        result = self.provider.format_tool_calls_for_execution(tool_calls)
        expected = [
            {
                "id": "call_1",
                "name": "test_tool",
                "input": {}
            }
        ]
        self.assertEqual(result, expected)
    
    def test_format_tool_results_for_conversation(self):
        """Test format_tool_results_for_conversation method."""
        tool_calls = [{"id": "call_1", "name": "test_tool"}]
        tool_outputs = [[
            {"type": "text", "text": "Result text"},
            {"type": "image", "source": {"type": "base64", "data": "image_data"}}
        ]]
        
        result, image_parts = self.provider.format_tool_results_for_conversation(tool_calls, tool_outputs)
        
        self.assertEqual(len(result), 3)  # tool_result + text + image
        self.assertEqual(result[0]["type"], "tool_result")
        self.assertEqual(result[0]["tool_use_id"], "call_1")
        self.assertEqual(result[0]["content"], "Result text")
        self.assertEqual(len(image_parts), 1)


class TestFactory(unittest.TestCase):
    """Test LLM provider factory."""
    
    def setUp(self):
        """Set up test environment."""
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_create_claude_provider(self):
        """Test creating Claude provider."""
        os.environ['ANTHROPIC_API_KEY'] = 'test_key'
        
        provider = create_llm_provider('claude-3-7-sonnet-latest')
        
        self.assertIsInstance(provider, ClaudeProvider)
        self.assertEqual(provider.provider_name, "Claude")
        self.assertEqual(provider.api_key, 'test_key')
        self.assertEqual(provider.model, 'claude-3-7-sonnet-latest')
    
    def test_create_gemini_provider(self):
        """Test creating Gemini provider."""
        os.environ['GEMINI_API_KEY'] = 'test_key'
        
        provider = create_llm_provider('gemini-2.5-flash')
        
        self.assertIsInstance(provider, GeminiProvider)
        self.assertEqual(provider.provider_name, "Gemini")
        self.assertEqual(provider.api_key, 'test_key')
        self.assertEqual(provider.model, 'gemini-2.5-flash')
    
    def test_create_provider_with_override_key(self):
        """Test creating provider with API key override."""
        provider = create_llm_provider('claude-3-7-sonnet-latest', api_key='override_key')
        
        self.assertIsInstance(provider, ClaudeProvider)
        self.assertEqual(provider.api_key, 'override_key')
    
    def test_create_provider_missing_key(self):
        """Test creating provider with missing API key."""
        # Clear the environment variable that was set by the test runner
        if 'ANTHROPIC_API_KEY' in os.environ:
            del os.environ['ANTHROPIC_API_KEY']
            
        with self.assertRaises(ValueError) as context:
            create_llm_provider('claude-3-7-sonnet-latest')
        
        self.assertIn("ANTHROPIC_API_KEY not found", str(context.exception))
    
    def test_create_provider_unsupported_model(self):
        """Test creating provider with unsupported model."""
        with self.assertRaises(ValueError) as context:
            create_llm_provider('unsupported-model')
        
        self.assertIn("Unsupported model", str(context.exception))


class TestClaudeProvider(unittest.TestCase):
    """Test Claude provider functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('llm_providers.claude_provider.anthropic.Anthropic'):
            self.provider = ClaudeProvider('test_key', 'claude-3-7-sonnet-latest')
    
    def test_initialization(self):
        """Test Claude provider initialization."""
        self.assertEqual(self.provider.provider_name, "Claude")
        self.assertTrue(self.provider.supports_thinking)
        self.assertEqual(self.provider.api_key, 'test_key')
        self.assertEqual(self.provider.model, 'claude-3-7-sonnet-latest')
    
    def test_format_tools(self):
        """Test Claude tools formatting."""
        tools = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}}
            }
        ]
        
        result = self.provider.format_tools(tools)
        expected = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}}
            }
        ]
        self.assertEqual(result, expected)
    
    def test_format_messages_basic(self):
        """Test Claude message formatting."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"}
        ]
        
        result = self.provider.format_messages(messages)
        
        # System message should be excluded
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[1]["role"], "assistant")
    
    def test_format_messages_with_tool_calls(self):
        """Test Claude message formatting with tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "I'll use a tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"param": "value"}'
                        }
                    }
                ]
            }
        ]
        
        result = self.provider.format_messages(messages)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "assistant")
        self.assertEqual(len(result[0]["content"]), 2)  # text + tool_use
        self.assertEqual(result[0]["content"][0]["type"], "text")
        self.assertEqual(result[0]["content"][1]["type"], "tool_use")
    
    def test_format_messages_tool_results(self):
        """Test Claude message formatting with tool results."""
        messages = [
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "Result"}
                ]
            }
        ]
        
        result = self.provider.format_messages(messages)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")
    
    def test_extract_system_message(self):
        """Test system message extraction."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"}
        ]
        
        result = self.provider._extract_system_message(messages)
        self.assertEqual(result, "System message")
    
    def test_extract_system_message_none(self):
        """Test system message extraction when none exists."""
        messages = [{"role": "user", "content": "User message"}]
        
        result = self.provider._extract_system_message(messages)
        self.assertIsNone(result)


class TestGeminiProvider(unittest.TestCase):
    """Test Gemini provider functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('llm_providers.gemini_provider.genai.Client'):
            self.provider = GeminiProvider('test_key', 'gemini-2.5-flash')
    
    def test_initialization(self):
        """Test Gemini provider initialization."""
        self.assertEqual(self.provider.provider_name, "Gemini")
        self.assertTrue(self.provider.supports_thinking)  # 2.5 model
        self.assertEqual(self.provider.api_key, 'test_key')
        self.assertEqual(self.provider.model, 'gemini-2.5-flash')
    
    def test_supports_thinking_false(self):
        """Test thinking support for non-2.5 models."""
        with patch('llm_providers.gemini_provider.genai.Client'):
            provider = GeminiProvider('test_key', 'gemini-1.5-pro')
            self.assertFalse(provider.supports_thinking)
    
    def test_format_tools(self):
        """Test Gemini tools formatting."""
        tools = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {"type": "object", "properties": {"param": {"type": "string"}}}
            }
        ]
        
        result = self.provider.format_tools(tools)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].function_declarations), 1)
        self.assertEqual(result[0].function_declarations[0].name, "test_tool")
    
    def test_count_images_in_messages(self):
        """Test image counting in messages."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Text"},
                    {"type": "image", "source": {"type": "base64", "data": "image1"}},
                    {"type": "image", "source": {"type": "base64", "data": "image2"}}
                ]
            }
        ]
        
        result = self.provider._count_images_in_messages(messages)
        self.assertEqual(result, 2)
    
    def test_format_tool_results_for_conversation(self):
        """Test Gemini tool results formatting."""
        tool_calls = [{"id": "call_1", "name": "test_tool"}]
        tool_outputs = [[
            {"type": "text", "text": '{"status": "success", "data": "test"}'},
            {"type": "image", "source": {"type": "base64", "data": "image_data"}}
        ]]
        
        result, image_parts = self.provider.format_tool_results_for_conversation(tool_calls, tool_outputs)
        
        self.assertEqual(len(result), 3)  # tool_result + text + image
        self.assertEqual(result[0]["type"], "tool_result")
        self.assertEqual(result[0]["tool_name"], "test_tool")
        self.assertIn("status", result[0]["content"])  # JSON should be formatted
        self.assertEqual(len(image_parts), 1)


class TestRetryLogic(unittest.IsolatedAsyncioTestCase):
    """Test retry logic functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('llm_providers.claude_provider.anthropic.Anthropic'):
            self.provider = ClaudeProvider('test_key', 'claude-3-7-sonnet-latest')
    
    @patch('asyncio.sleep', new_callable=AsyncMock)  # Mock sleep to speed up tests
    async def test_retry_with_server_error(self, mock_sleep):
        """Test retry logic with server overload error."""
        # Create a mock that fails twice then succeeds
        call_count = 0
        
        async def mock_generate_response_impl(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:
                raise Exception("Server overload - please retry")
            else:
                return LLMResponse(
                    content="Success after retry!",
                    provider="Test",
                    usage={"total_tokens": 10}
                )
        
        with patch.object(self.provider, '_generate_response_impl', side_effect=mock_generate_response_impl):
            response = await self.provider.generate_response([
                {'role': 'user', 'content': 'Test message'}
            ])
            
            self.assertEqual(response.content, "Success after retry!")
            self.assertEqual(call_count, 3)  # Should have made 3 attempts
            
            # Verify sleep was called for retries
            self.assertEqual(mock_sleep.call_count, 2)  # 2 retries = 2 sleeps
            mock_sleep.assert_any_call(1.0)  # First retry delay
            mock_sleep.assert_any_call(2.0)  # Second retry delay
    
    async def test_retry_with_non_retryable_error(self):
        """Test that non-retryable errors are not retried."""
        call_count = 0
        
        async def mock_generate_response_impl(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid API key")
        
        with patch.object(self.provider, '_generate_response_impl', side_effect=mock_generate_response_impl):
            with self.assertRaises(ValueError) as context:
                await self.provider.generate_response([
                    {'role': 'user', 'content': 'Test message'}
                ])
            
            self.assertIn("Invalid API key", str(context.exception))
            self.assertEqual(call_count, 1)  # Should have made only 1 attempt
    
    @patch('asyncio.sleep', new_callable=AsyncMock)  # Mock sleep to speed up tests
    async def test_retry_exhaustion(self, mock_sleep):
        """Test that retry logic gives up after max attempts."""
        call_count = 0
        
        async def mock_generate_response_impl(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("Rate limit exceeded")
        
        with patch.object(self.provider, '_generate_response_impl', side_effect=mock_generate_response_impl):
            with self.assertRaises(Exception) as context:
                await self.provider.generate_response([
                    {'role': 'user', 'content': 'Test message'}
                ])
            
            self.assertIn("Rate limit exceeded", str(context.exception))
            self.assertEqual(call_count, 6)  # Should have made 6 attempts (1 + 5 retries)
            
            # Verify sleep was called for all retries
            self.assertEqual(mock_sleep.call_count, 5)  # 5 retries = 5 sleeps
            mock_sleep.assert_any_call(1.0)  # First retry delay
            mock_sleep.assert_any_call(2.0)  # Second retry delay
            mock_sleep.assert_any_call(4.0)  # Third retry delay (exponential backoff)
    
    async def test_retry_with_different_error_types(self):
        """Test retry logic with different types of retryable errors."""
        retryable_errors = [
            "Server overload",
            "Rate limit exceeded", 
            "Service unavailable",
            "Internal server error",
            "Connection timeout",
            "Resource exhausted",
            "Too many requests",
            "Server is busy"
        ]
        
        for error_msg in retryable_errors:
            with self.subTest(error=error_msg):
                call_count = 0
                
                async def mock_generate_response_impl(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    
                    if call_count == 1:
                        raise Exception(error_msg)
                    else:
                        return LLMResponse(content="Success", provider="Test")
                
                with patch.object(self.provider, '_generate_response_impl', side_effect=mock_generate_response_impl):
                    with patch('asyncio.sleep', new_callable=AsyncMock):
                        response = await self.provider.generate_response([
                            {'role': 'user', 'content': 'Test message'}
                        ])
                        
                        self.assertEqual(response.content, "Success")
                        self.assertEqual(call_count, 2)  # Should retry once
    
    def test_retry_decorator_parameters(self):
        """Test that retry decorator can be configured with different parameters."""
        from llm_providers.base_provider import retry_llm_call
        
        # Test custom retry parameters
        @retry_llm_call(max_retries=2, initial_delay=0.5)
        async def test_function():
            raise Exception("Test error")
        
        # This test just verifies the decorator can be applied with custom parameters
        self.assertTrue(callable(test_function))


if __name__ == '__main__':
    unittest.main() 