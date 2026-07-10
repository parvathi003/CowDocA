import chromadb
from rag.embedding import get_embedding

# ==========================================================
# Connect to ChromaDB
# ==========================================================

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="cattle_disease_knowledge"
)

# ==========================================================
# Disease Mapping
# ==========================================================

DISEASE_MAPPING = {

    "foot and mouth disease": "FMD",
    "fmd": "FMD",

    "lumpy skin disease": "LSD",
    "lsd": "LSD",

    "mastitis": "Mastitis",

    "foot rot": "FootRot",
    "footrot": "FootRot",

    # Ringworm information is stored inside the General PDF
    "ringworm": "General"
}


# ==========================================================
# Detect Disease Name
# ==========================================================

def detect_disease(question):

    question = question.lower()

    for keyword, disease in DISEASE_MAPPING.items():

        if keyword in question:
            return disease

    return None


# ==========================================================
# Retrieve Documents
# ==========================================================

def retrieve_documents(question, top_k=5):

    detected_disease = detect_disease(question)

    query_embedding = get_embedding(question)

    # ------------------------------------------------------
    # Disease Specific Search
    # ------------------------------------------------------

    if detected_disease:

        results = collection.query(

            query_embeddings=[query_embedding],

            n_results=top_k,

            where={
                "disease": detected_disease
            }

        )

    # ------------------------------------------------------
    # Search Entire Knowledge Base
    # ------------------------------------------------------

    else:

        results = collection.query(

            query_embeddings=[query_embedding],

            n_results=top_k

        )

    # ------------------------------------------------------
    # Empty Results
    # ------------------------------------------------------

    if (
        len(results["documents"]) == 0
        or len(results["documents"][0]) == 0
    ):

        return {

            "documents": [],

            "sources": [],

            "distances": [],

            "detected_disease": detected_disease

        }

    # ------------------------------------------------------
    # Filter Low Quality Results
    # ------------------------------------------------------

    filtered_documents = []
    filtered_sources = []
    filtered_distances = []

    for doc, meta, distance in zip(

        results["documents"][0],

        results["metadatas"][0],

        results["distances"][0]

    ):

        # Lower distance = Better Match
        # 0.80 is a reasonable starting threshold
        if distance <= 0.80:

            filtered_documents.append(doc)

            filtered_sources.append(meta)

            filtered_distances.append(distance)

    # ------------------------------------------------------
    # If Filtering Removed Everything,
    # Return Original Results
    # ------------------------------------------------------

    if len(filtered_documents) == 0:

        filtered_documents = results["documents"][0]

        filtered_sources = results["metadatas"][0]

        filtered_distances = results["distances"][0]

    return {

        "documents": filtered_documents,

        "sources": filtered_sources,

        "distances": filtered_distances,

        "detected_disease": detected_disease

    }


# ==========================================================
# Test Retrieval
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CowDoc AI - Knowledge Base Search")
    print("=" * 60)

    while True:

        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            break

        results = retrieve_documents(question)

        if len(results["documents"]) == 0:

            print("\nNo relevant information found.\n")
            continue

        print("\nRetrieved Documents\n")

        for i, (doc, meta, distance) in enumerate(

            zip(

                results["documents"],

                results["sources"],

                results["distances"]

            ),

            start=1

        ):

            print("=" * 80)

            print(f"Result : {i}")

            print(f"Disease : {meta['disease']}")

            print(f"Source : {meta['source']}")

            print(f"Distance : {round(distance,4)}")

            print("-" * 80)

            print(doc[:700])

            print()