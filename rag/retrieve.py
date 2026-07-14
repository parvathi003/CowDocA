"""
RAG Retrieval Module
====================

Responsibilities
----------------
- Receive the disease identified by Stage 1
- Receive the normalized query from Stage 1
- Search only within that disease
- Return the most relevant knowledge chunks

This module NEVER:
- Detects diseases
- Calls the LLM
- Answers questions
"""

import chromadb

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    RAG_TOP_K
)

from rag.embedding import get_embedding

# ==========================================================
# Connect to ChromaDB
# ==========================================================

client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

collection = client.get_collection(
    name=CHROMA_COLLECTION
)

# ==========================================================
# Retrieve Documents
# ==========================================================

def retrieve_documents(
    disease,
    question,
    top_k=RAG_TOP_K
):
    """
    Retrieve the most relevant chunks for the
    identified disease.
    """

    query_embedding = get_embedding(question)

    try:

        results = collection.query(

            query_embeddings=[query_embedding],

            n_results=top_k,

            where={
                "disease": disease
            }

        )

    except Exception:

        return {

            "documents": [],

            "sources": [],

            "distances": []

        }

    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])

    return {

        "documents": documents[0] if documents else [],

        "sources": metadatas[0] if metadatas else [],

        "distances": distances[0] if distances else []

    }


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    while True:

        disease = input("\nDisease: ")

        if disease.lower() == "exit":
            break

        question = input("Question: ")

        results = retrieve_documents(

            disease=disease,

            question=question

        )

        if not results["documents"]:

            print("\nNo documents found.\n")

            continue

        print("\nRetrieved Documents\n")

        for i, (

            document,

            metadata,

            distance

        ) in enumerate(

            zip(

                results["documents"],

                results["sources"],

                results["distances"]

            ),

            start=1

        ):

            print("=" * 80)

            print(f"Result : {i}")

            print(f"Disease : {metadata.get('disease','Unknown')}")

            print(f"Source : {metadata.get('source','Unknown')}")

            print(f"Distance : {round(distance,4)}")

            print("-" * 80)

            print(document[:600])

            print()