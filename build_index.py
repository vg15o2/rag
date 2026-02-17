"""Build ChromaDB vector index from file_info.txt."""

import os
from dotenv import load_dotenv

load_dotenv()

import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

FILE_INFO_PATH = os.path.join(os.path.dirname(__file__), "file_info.txt")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "file_info"


def main():
    # Read file_info.txt and split into per-file entries
    with open(FILE_INFO_PATH) as f:
        content = f.read()

    entries = [e.strip() for e in content.split("\n\n") if e.strip()]
    print(f"Found {len(entries)} entries in file_info.txt")

    # Extract filename from each entry as metadata
    documents = []
    metadatas = []
    ids = []
    for entry in entries:
        # First line contains "File: x.csv" or starts with filename
        for word in entry.split():
            if word.endswith(".csv"):
                filename = word.rstrip(".,;:")
                break
        documents.append(entry)
        metadatas.append({"filename": filename})
        ids.append(filename)
        print(f"  {filename}: {entry[:80]}...")

    # Create embeddings and store in ChromaDB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Remove old index if exists
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print(f"\nRemoved old index at {CHROMA_DIR}")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    vectorstore.add_texts(texts=documents, metadatas=metadatas, ids=ids)

    print(f"\nCreated ChromaDB index at {CHROMA_DIR}")
    print(f"Indexed {len(documents)} documents")

    # Quick test
    results = vectorstore.similarity_search("Bitcoin", k=2)
    print(f"\nTest query 'Bitcoin' returned:")
    for r in results:
        print(f"  - {r.metadata['filename']}: {r.page_content[:60]}...")


if __name__ == "__main__":
    main()
