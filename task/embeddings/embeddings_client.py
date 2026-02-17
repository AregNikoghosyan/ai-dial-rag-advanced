import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'

class DialEmbeddingsClient:

    def __init__(self, deployment_name: str, api_key: str):
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.url = DIAL_EMBEDDINGS.format(model=self.deployment_name)

    def get_embeddings(self, inputs: list[str]) -> dict[int, list[float]]:
        """
        Generates embeddings for a list of input texts.

        :param inputs: List of text strings
        :return: dict {index: embedding_vector}
        """

        headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "input": inputs
        }

        response = requests.post(
            self.url,
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(
                f"Embeddings request failed: {response.status_code} - {response.text}"
            )

        response_json = response.json()

        embeddings = {}

        for item in response_json.get("data", []):
            index = item["index"]
            vector = item["embedding"]

            if len(vector) != 768:
                raise ValueError(
                f"Embedding dimension mismatch. Expected 768, got {len(vector)}"
    )


            embeddings[index] = vector

        return embeddings
