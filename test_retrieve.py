from rag.retrieve import retrieve_documents

print("=" * 60)
print("CowDoc AI - Knowledge Base Search")
print("=" * 60)

question = input("\nAsk a question: ")

results = retrieve_documents(question)

print("\nRetrieved Documents\n")

for i in range(len(results["documents"])):

    print("=" * 80)

    print("Result :", i + 1)

    print("Source :", results["sources"][i]["source"])

    print("Disease :", results["sources"][i]["disease"])

    print("Distance :", round(results["distances"][i], 4))

    print("-" * 80)

    print(results["documents"][i])

    print()