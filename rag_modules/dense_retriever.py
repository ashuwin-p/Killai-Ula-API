"""
    Author: Ashuwin P
    Email: ashuwin2210335@ssn.edu.in
    File Description:
    This module defines a DenseRetriever class that wraps interactions with a ChromaDB
    persistent vector database for embedding-based retrieval. It initializes a ChromaDB client,
    creates or gets a named collection with an embedding function, and supports document
    ingestion, querying by text, and counting stored documents.

    Usage:
    - Instantiate DenseRetriever with optional database path, collection name, and embedding model.
    - Use ingest(documents, metadatas, ids) to add documents with metadata and IDs.
    - Use query(query, k) to retrieve top-k most similar documents for a text query.
    - Use count() to get the number of documents in the collection.

    The class uses the SentenceTransformerEmbeddingFunction to create embeddings,
    defaulting to the "all-MiniLM-L6-v2" model.
"""

import os
import json
import chromadb
from chromadb.utils import embedding_functions


class DenseRetriever:
    """
    DenseRetriever manages a persistent vector database for semantic document retrieval
    using ChromaDB and sentence transformer embeddings.

    Attributes:
        client (chromadb.PersistentClient): Persistent ChromaDB client instance connected to disk.
        embed_fn (SentenceTransformerEmbeddingFunction): Embedding function using a pre-trained model.
        collection (Collection): ChromaDB collection for storing documents and embeddings.
    """

    def __init__(
        self,
        db_path: str = "./tourism_advanced_db",
        collection_name: str = "tourism_advanced",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize DenseRetriever by creating a persistent client, embedding function,
        and getting or creating the named collection.

        Args:
            db_path (str): Filesystem path to store the persistent ChromaDB database.
            collection_name (str): Name of the collection to create or access.
            model_name (str): Sentence transformer model to use for embedding text.
        """
        # Persistent client stores data on disk at db_path
        self.client = chromadb.PersistentClient(path=db_path)

        # Embedding function using a sentence transformer model
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        # Get existing or create new collection associated with this embedding function
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    def ingest(self, documents, metadatas, ids):
        """
        Add new documents along with their metadata and unique IDs to the collection.

        Args:
            documents (list[str]): List of text documents to add.
            metadatas (list[dict]): List of metadata dictionaries for each document.
            ids (list[str]): Unique string IDs identifying each document in the collection.
        """
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query, k: int = 10):
        """
        Query the collection for the top-k documents most similar to the input text query.

        Args:
            query (str): The text query to embed and search against the collection.
            k (int): Number of top results to return (default 10).

        Returns:
            dict: Query results containing matched documents, metadata, distances, etc.
        """
        return self.collection.query(
            query_texts=[query],
            n_results=k
        )

    def count(self):
        """
        Count the number of documents currently stored in the collection.

        Returns:
            int: Total document count in the collection.
        """
        return self.collection.count()
    
    # ---------------------------------------------------------
    #                    SAVE & LOAD METHODS
    # ---------------------------------------------------------
    def save(self, export_path: str = "./dense_export.json"):
        """
        Export all data from ChromaDB into a portable JSON file.

        Args:
            export_path (str): File path to save export data.
        """
        all_ids = self.collection.get()["ids"]

        # Chroma may paginate, so fetch in chunks
        CHUNK = 200
        full_export = []

        for i in range(0, len(all_ids), CHUNK):
            batch_ids = all_ids[i:i + CHUNK]
            batch = self.collection.get(ids=batch_ids)
            full_export.append(batch)

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(full_export, f, indent=2)

        print(f"[DenseRetriever] Exported {len(all_ids)} documents → {export_path}")

    def load(self, import_path: str = "./dense_export.json"):
        """
        Load previously exported data into the current collection.

        This replaces existing data.

        Args:
            import_path (str): JSON export file.
        """
        if not os.path.exists(import_path):
            raise FileNotFoundError(f"File not found: {import_path}")

        # Clear old data
        existing_ids = self.collection.get()["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        # Load JSON
        with open(import_path, "r", encoding="utf-8") as f:
            batches = json.load(f)

        # Re-insert
        for batch in batches:
            self.collection.add(
                documents=batch["documents"],
                metadatas=batch["metadatas"],
                ids=batch["ids"]
            )

        print(f"[DenseRetriever] Imported {len(existing_ids)} → {import_path}")