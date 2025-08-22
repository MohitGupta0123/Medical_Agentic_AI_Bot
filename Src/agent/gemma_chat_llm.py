# src/agent/gemma_chat_llm.py
from langchain_core.language_models import LLM
from huggingface_hub import InferenceClient
from pydantic import PrivateAttr


class GemmaChatLLM(LLM):
    model: str = "google/gemma-3-27b-it"
    temperature: float = 0.3
    max_tokens: int = 512
    hf_token: str = None  # new: added for user-provided token

    _client: InferenceClient = PrivateAttr()

    def __init__(self, model: str = None, temperature: float = 0.3, max_tokens: int = 512, hf_token: str = None):
        super().__init__(model=model or self.model, temperature=temperature, max_tokens=max_tokens)
        self.hf_token = hf_token
        self._client = InferenceClient(token=hf_token, provider="auto", timeout=400)  # token added

    @property
    def _llm_type(self) -> str:
        return "gemma-chat-llm"

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        """Send prompt to Hugging Face Gemma with HF token."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message["content"]


class GemmaChatLLM2(LLM):
    model: str = "google/gemma-3-27b-it"
    temperature: float = 0.3
    max_tokens: int = 512
    hf_token: str = None  # new: added for user-provided token

    _client: InferenceClient = PrivateAttr()

    def __init__(self, model: str = None, temperature: float = 0, max_tokens: int = 50, hf_token: str = None):
        model = model or "google/gemma-3-27b-it"
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.hf_token = hf_token
        self._client = InferenceClient(token=hf_token, provider="auto", timeout=400)


    @property
    def _llm_type(self) -> str:
        return "gemma-chat-llm"

    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        """Send prompt to Hugging Face Gemma with HF token."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful medical assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message["content"]
