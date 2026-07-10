from rag.embedding import get_embedding

text = "My cow has fever"

embedding = get_embedding(text)

print(type(embedding))
print(len(embedding))
