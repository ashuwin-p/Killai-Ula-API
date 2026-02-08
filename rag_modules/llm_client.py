import os
import time
import re

class LLMClient:
    """Abstract Base Class for LLMs"""
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class GroqClient(LLMClient):
    def __init__(self, api_key: str = None, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        try:
            from groq import Groq, RateLimitError
            self.Groq = Groq
            self.RateLimitError = RateLimitError
        except ImportError:
            raise ImportError("pip install groq")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found.")
        
        self.client = self.Groq(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        retries = 3
        for attempt in range(retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    temperature=0.3,
                )
                return chat_completion.choices[0].message.content
            except self.RateLimitError as e:
                wait_time = float(re.search(r"(\d+(\.\d+)?)s", str(e)).group(1)) + 2
                time.sleep(wait_time)
            except Exception as e:
                if attempt == retries - 1: return f"Error: {e}"
                time.sleep(2)
        return "Error: Timeout"

# Example: Easy to add OpenAI later
class OpenAIClient(LLMClient):
    def __init__(self, api_key=None):
        # Initialize OpenAI client here
        pass
    def generate(self, prompt: str) -> str:
        # Call OpenAI API here
        return "OpenAI Response"