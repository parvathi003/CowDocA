import os
import fitz
import chromadb

from rag.embedding import get_embedding

# ==========================================
# Configuration
# ==========================================

PDF_FOLDER = "knowledge_base"
DB_PATH = "chroma_db"
COLLECTION_NAME = "cattle_disease_knowledge"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ==========================================
# Connect to ChromaDB
# ==========================================

client = chromadb.PersistentClient(path=DB_PATH)

try:
    client.delete_collection(COLLECTION_NAME)
    print("Old knowledge base deleted.")
except:
    print("Creating new knowledge base...")

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# ==========================================
# Read PDF
# ==========================================

def extract_text(pdf_path):

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


# ==========================================
# Chunk Text
# ==========================================

def chunk_text(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if len(chunk) > 80:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ==========================================
# Build Knowledge Base
# ==========================================

total_pdf = 0
total_chunks = 0

print("=" * 60)
print("Building CowDoc AI Knowledge Base")
print("=" * 60)

for disease_folder in os.listdir(PDF_FOLDER):

    folder_path = os.path.join(PDF_FOLDER, disease_folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nDisease : {disease_folder}")

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(".pdf"):
            continue

        # Skip Annual Report
        if filename == "NDDB_AR_2023_24.pdf":
            print("Skipping NDDB Annual Report")
            continue

        pdf_path = os.path.join(folder_path, filename)

        print(f"Processing : {filename}")

        text = extract_text(pdf_path)

        print(f"Characters : {len(text)}")

        chunks = chunk_text(text)

        print(f"Chunks : {len(chunks)}")

        for chunk_id, chunk in enumerate(chunks):

            embedding = get_embedding(chunk)

            collection.add(
                ids=[f"{disease_folder}_{filename}_{chunk_id}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": filename,
                    "disease": disease_folder,
                    "chunk_id": chunk_id
                }]
            )

            total_chunks += 1

        total_pdf += 1

print("\n" + "=" * 60)
print("Knowledge Base Created Successfully")
print("=" * 60)
print(f"PDFs Processed : {total_pdf}")
print(f"Chunks Created : {total_chunks}")
print("=" * 60)