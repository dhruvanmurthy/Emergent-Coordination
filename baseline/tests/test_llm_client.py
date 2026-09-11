import unittest

from llm_client import (
    _call_chat_completion,
    _normalize_azure_endpoint,
    _uses_max_completion_tokens,
)


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return {"ok": True}


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


class LLMClientTests(unittest.TestCase):
    def test_normalize_azure_endpoint_removes_openai_suffixes(self):
        self.assertEqual(
            _normalize_azure_endpoint("https://example.openai.azure.com/openai/v1/"),
            "https://example.openai.azure.com",
        )
        self.assertEqual(
            _normalize_azure_endpoint("https://example.openai.azure.com/openai"),
            "https://example.openai.azure.com",
        )

    def test_gpt5_uses_completion_token_parameter_in_json_mode(self):
        client = _FakeClient()

        _call_chat_completion(client, "gpt-5-mini", "choose", 64)

        kwargs = client.chat.completions.kwargs
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["max_completion_tokens"], 64)
        self.assertNotIn("max_tokens", kwargs)
        self.assertTrue(_uses_max_completion_tokens("gpt-5-mini"))

    def test_non_reasoning_model_uses_max_tokens(self):
        client = _FakeClient()

        _call_chat_completion(client, "gpt-4o-mini", "choose", 64)

        kwargs = client.chat.completions.kwargs
        self.assertEqual(kwargs["max_tokens"], 64)
        self.assertNotIn("max_completion_tokens", kwargs)
        self.assertFalse(_uses_max_completion_tokens("gpt-4o-mini"))


if __name__ == "__main__":
    unittest.main()