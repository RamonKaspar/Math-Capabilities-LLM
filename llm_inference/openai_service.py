import os
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import List

from .llm_interface import LLMInterface

class OpenAIService(LLMInterface):
    def __init__(self, model_name, temperature, max_tokens):
        super().__init__(model_name, temperature, max_tokens)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def make_request(self, messages: List[ChatCompletionMessageParam]) -> tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        response_message = response.choices[0].message.content
        completion_tokens_used = response.usage.completion_tokens
        prompt_tokens_used = response.usage.prompt_tokens
        return response_message, prompt_tokens_used, completion_tokens_used
