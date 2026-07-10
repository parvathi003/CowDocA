from rag.retrieve import retrieve_documents

print("=" * 70)
print("CowDoc AI - RAG System Test")
print("=" * 70)

test_questions = [

    "What are the symptoms of Foot and Mouth Disease?",

    "How can I prevent Lumpy Skin Disease?",

    "What causes Mastitis?",

    "What is Foot Rot?",

    "How is Ringworm treated?",

    "What vaccine is used for FMD?"

]

for question in test_questions:

    print("\n" + "="*70)
    print("QUESTION:")
    print(question)

    results = retrieve_documents(question)

    print("\nTOP RETRIEVED DOCUMENTS")

    for i in range(len(results["documents"])):

        print("-"*60)

        print("Rank:", i+1)

        print("Disease:",
              results["sources"][i]["disease"])

        print("Source:",
              results["sources"][i]["source"])

        print("Distance:",
              round(results["distances"][i],4))