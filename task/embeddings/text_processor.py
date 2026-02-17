from enum import StrEnum
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"
    COSINE_DISTANCE = "cosine"


class TextProcessor:

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(
        self,
        file_path: str,
        chunk_size: int = 300,
        overlap: int = 50,
        dimensions: int = 768,
        truncate: bool = True
    ):
        with self._get_connection() as conn:
            with conn.cursor() as cursor:

                if truncate:
                    cursor.execute("TRUNCATE TABLE vectors;")

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                embeddings_dict = self.embeddings_client.get_embeddings(chunks)
                document_name = os.path.basename(file_path)

                for index, chunk in enumerate(chunks):
                    embedding = embeddings_dict[index]

                    if len(embedding) != dimensions:
                        raise ValueError(
                            f"Embedding dimension mismatch. Expected {dimensions}, got {len(embedding)}"
                        )

                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                    cursor.execute(
                        """
                        INSERT INTO vectors (document_name, text, embedding)
                        VALUES (%s, %s, %s::vector)
                        """,
                        (
                            document_name,
                            chunk,
                            embedding_str
                        )
                    )

            conn.commit()

    def search(
        self,
        search_mode: SearchMode,
        user_request: str,
        top_k: int = 5,
        min_score: float = 0.6,
        dimensions: int = 768
    ):
        embedding_dict = self.embeddings_client.get_embeddings([user_request])
        query_embedding = embedding_dict[0]

        if len(query_embedding) != dimensions:
            raise ValueError(
                f"Query embedding dimension mismatch. Expected {dimensions}, got {len(query_embedding)}"
            )

        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        operator = "<=>" if search_mode == SearchMode.COSINE_DISTANCE else "<->"

        if search_mode == SearchMode.COSINE_DISTANCE:
            query = f"""
                SELECT text,
                       embedding {operator} %s::vector AS distance
                FROM vectors
                ORDER BY embedding {operator} %s::vector
                LIMIT %s;
            """
            params = (embedding_str, embedding_str, top_k)
        else:
            query = f"""
                SELECT text,
                       embedding {operator} %s::vector AS distance
                FROM vectors
                WHERE embedding {operator} %s::vector <= %s
                ORDER BY embedding {operator} %s::vector
                LIMIT %s;
            """
            params = (embedding_str, embedding_str, min_score, embedding_str, top_k)

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()

        return [row["text"] for row in results]
