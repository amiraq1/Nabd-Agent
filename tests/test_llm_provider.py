import os
import unittest
from unittest.mock import patch

from nabd.llm import LLMClient, LLMError


class LLMProviderTests(unittest.TestCase):
    def test_auto_prefers_nvidia_when_key_exists(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test", "OPENAI_API_KEY": "", "GEMINI_API_KEY": ""}, clear=False):
            client = LLMClient("auto")
            self.assertEqual(client.provider, "nvidia")
            self.assertEqual(client.model, "meta/llama-3.1-8b-instruct")

    def test_explicit_nvidia_uses_configured_model_and_url(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test", "NABD_NVIDIA_MODEL": "meta/test-model"}, clear=False):
            client = LLMClient("nvidia")
            self.assertEqual(client.provider, "nvidia")
            self.assertEqual(client.model, "meta/test-model")

    def test_invalid_provider_is_rejected(self):
        with self.assertRaises(LLMError):
            LLMClient("unknown")


if __name__ == "__main__":
    unittest.main()
