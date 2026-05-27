import requests


class OllamaClient:

    def __init__(
        self,
        model_name: str,
        url: str
    ):

        self.model_name = model_name
        self.url = url

    def generate(
        self,
        query: str,
        context: str
    ) -> str:

        prompt = f"""
You are a factual QA assistant.

Rules:
- Use ONLY the provided context
- If answer is unavailable, say "Not found"
- Keep answers concise

Context:
{context}

Question:
{query}
"""

        try:

            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )

            response.raise_for_status()

            return response.json()["response"]

        except requests.RequestException as error:

            return f"LLM request failed: {error}"
