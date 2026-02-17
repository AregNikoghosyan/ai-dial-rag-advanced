from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


SYSTEM_PROMPT = """
You are a RAG-powered Microwave Manual Assistant.

You will receive:
1) RAG Context - retrieved parts of the microwave manual.
2) User Question.

Answer only using the provided RAG Context.
If the answer is not found in the context, respond:
"I cannot find this information in the manual."
Do not use outside knowledge.
Do not answer questions unrelated to microwave usage.
Keep answers clear and precise.
"""


USER_PROMPT = """
RAG Context:
{context}

User Question:
{question}
"""


def main():

    embeddings_client = DialEmbeddingsClient(
        deployment_name="text-embedding-005",
        api_key=API_KEY
    )

    chat_client = DialChatCompletionClient(
    deployment_name="anthropic.claude-haiku-4-5-20251001-v1:0",
    api_key=API_KEY
)


    db_config = {
        "host": "localhost",
        "port": 5433,
        "database": "vectordb",
        "user": "postgres",
        "password": "postgres"
    }

    processor = TextProcessor(embeddings_client, db_config)

    processor.process_text_file(
        "task/embeddings/microwave_manual.txt"
    )

    print("Microwave RAG Assistant is ready.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        retrieved_chunks = processor.search(
            search_mode=SearchMode.COSINE_DISTANCE,
            user_request=user_input,
            top_k=5,
            min_score=0.6
        )

        if not retrieved_chunks:
            print("Assistant: I cannot find this information in the manual.\n")
            continue

        context = "\n\n".join(retrieved_chunks)

        formatted_user_prompt = USER_PROMPT.format(
            context=context,
            question=user_input
        )

        conversation = Conversation()
        conversation.add_message(Message(role=Role.SYSTEM, content=SYSTEM_PROMPT))
        conversation.add_message(Message(role=Role.USER, content=formatted_user_prompt))

        response = chat_client.get_completion(conversation.messages)

        print(f"Assistant: {response.content}\n")


if __name__ == "__main__":
    main()
