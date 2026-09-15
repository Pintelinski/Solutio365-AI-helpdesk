"""
Loads AI-classification-setup/knowledge_base.json into a local Chroma
collection. Re-run this any time the knowledge base file changes - it
replaces the collection contents each time, so it's safe to run repeatedly.

Usage:
    python seed_chroma.py
"""
import json
from pathlib import Path
import os
from dotenv import load_dotenv

import chromadb
import ollama

load_dotenv() 

EMBED_MODEL = "nomic-embed-text"
OLLAMA_HOST = os.getenv("OLLAMA_HOST")  
CHROMA_PATH = Path(__file__).parent / "chroma_db"
KNOWLEDGE_BASE_PATH = Path(__file__).parent / "AI-classification-setup" / "knowledge_base.json"

ollama_client = ollama.Client(host=OLLAMA_HOST)


def main():
    entries = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(entries)} entries from {KNOWLEDGE_BASE_PATH}")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    client.delete_collection(name="known_issues") if "known_issues" in [c.name for c in client.list_collections()] else None
    collection = client.create_collection(name="known_issues")

    for entry in entries:
        embedding = ollama_client.embeddings(model=EMBED_MODEL, prompt=entry["description"])["embedding"]
        collection.add(
            ids=[entry["id"]],
            embeddings=[embedding],
            documents=[entry["description"]],
            metadatas=[{"category": entry["category"], "guidance": entry["guidance"]}],
        )
        print(f"  Added: {entry['id']}")

    print(f"Done. Collection now has {collection.count()} entries at {CHROMA_PATH}")


if __name__ == "__main__":
    main()