
import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import os

load_dotenv()

_chroma_clients = {}

def get_chroma_client(chroma_dir: str):
    path = str(Path(chroma_dir).resolve())

    if path not in _chroma_clients:

        _chroma_clients[path] = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False)
        )

    return _chroma_clients[path]


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories

    # TODO: Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [
        path for path in current_dir.iterdir()
        if path.is_dir()
        and (
            "chroma" in path.name.lower()
            or (path / "chroma.sqlite3").exists()
        )
    ]
    # TODO: Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # TODO: Wrap connection attempt in try-except block for error handling
        try:
            # TODO: Initialize database client with directory path and configuration settings
            client = get_chroma_client(str(chroma_dir))
            # client = chromadb.PersistentClient(
            #     path=str(chroma_dir),
            #     settings=Settings(
            #         anonymized_telemetry=False,  # Disable telemetry for privacy
            #         allow_reset=True             # Allow database reset for development
            #         )
            # )
            # TODO: Retrieve list of available collections from the database
            collections = client.list_collections()
            # TODO: Loop through each collection found
            for collection in collections:
                # TODO: Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}:{collection.name}"
                # TODO: Build information dictionary containing:
                try:
                    count = collection.count()
                except Exception:
                    count = "Unknown"
                    # TODO: Store directory path as string
                    # TODO: Store collection name
                    # TODO: Create user-friendly display name
                    # TODO: Get document count with fallback for unsupported operations
                # TODO: Add collection information to backends dictionary
                backends[key] = {
                    "path": str(chroma_dir.resolve()),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} / {collection.name} ({count} docs)",
                    "count": str(count)
                }
        
        # TODO: Handle connection or access errors gracefully
            # TODO: Create fallback entry for inaccessible directories
            # TODO: Include error information in display name with truncation
            # TODO: Set appropriate fallback values for missing information
        except Exception as e:
            error_text = str(e)[:80]

            key = f"{chroma_dir.name}:error"
            backends[key] = {
                "path": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} - Error: {error_text}",
                "count": "0"
            }
    # TODO: Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # TODO: Create a chomadb persistentclient

    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    client = get_chroma_client(chroma_dir)
    collection = client.get_collection(name=collection_name,embedding_function=embedding_function)
    # TODO: Return the collection with the collection_name
    return collection, True, None



def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # TODO: Initialize filter variable to None (represents no filtering)
    where_filter = None

    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs
    if mission_filter and mission_filter.lower() not in ["all", "none", ""]:
        where_filter = {
            "mission": mission_filter
        }
    # TODO: Execute database query with the following parameters:
        # TODO: Pass search query in the required format
        # TODO: Set maximum number of results to return
        # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )
    # TODO: Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # TODO: Initialize list with header text for context section
    context_parts = ["Retrieved Context Documents:\n"]

    # TODO: Loop through paired documents and their metadata using enumeration
    for i, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        # TODO: Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "unknown")
        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").title()

        # TODO: Extract source information from metadata with fallback value  
        source = metadata.get("source", "unknown source")
        # TODO: Extract category information from metadata with fallback value
        category = metadata.get("category", "general")
        # TODO: Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").title()
        
        # TODO: Create formatted source header with index number and extracted information
        source_header = (
            f"\n--- Source {i} ---\n"
            f"Mission: {mission}\n"
            f"Source: {source}\n"
            f"Category: {category}\n"
        )
        # TODO: Add source header to context parts list
        context_parts.append(source_header)
        
        # TODO: Check document length and truncate if necessary
        max_chars = 2500
        # TODO: Add truncated or full document content to context parts list
        if len(doc) > max_chars:
            doc = doc[:max_chars] + "\n...[truncated]"
        
        context_parts.append(doc)

    # TODO: Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)