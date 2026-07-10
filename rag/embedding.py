from sentence_transformers import SentenceTransformer

# Load the embedding model only once
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def get_embedding(text):
    """
    Convert text into an embedding vector.
    """
    return embedding_model.encode(text).tolist()
