import chromadb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
collection = client.get_or_create_collection("crud_demo")

# Start from scratch so that demo should be repeatable

try:
    collection.delete(ids=collection.get()["ids"])
except Exception:
    pass


# -------------------------------------------------
# Create
# -------------------------------------------------
documents = [
    "Python is a popular programming language for AI and data science.",
    "Machine learning enables computers to learn from data.",
    "ChromaDB is a vector database used in Retrieval-Augmented Generation (RAG).",
]

ids = [str(i) for i in range(1, len(documents) + 1)]
print(ids)

metadatas = [{"topic": "programming"}, {"topic": "ai"}, {"topic": "database"}]

collection.add(ids=ids, documents=documents, metadatas=metadatas)

print(f"Created {len(documents)} documents.\n")

# -------------------------------------------------
# Read
# -------------------------------------------------
result = collection.get(ids=['2'])
print("read by '2'")
print(f" Documents : {result['documents'][0]}")
print(f" Metadatas : {result['metadatas'][0]}")


# -------------------------------------------------
# Update
# -------------------------------------------------
collection.update(
    ids=['2'],
    documents=["Machine learning enables computers to learn patterns from data without explicit programming."],
    metadatas=[{ "topic": "ai", "updated": True }]
)

updated = collection.get(ids=["2"])
print("Updated document '2':")
print(f"  Document: {updated['documents'][0]}")
print(f"  Metadata: {updated['metadatas'][0]}\n")

# -------------------------------------------------
# Delete
# -------------------------------------------------
collection.delete(ids=["3"])
print("Deleted document '3'.")
print(f"Remaining document count: {collection.count()}")

