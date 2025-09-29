import os
import chromadb
from neo4j import GraphDatabase
from flask import Flask, request, jsonify
from waitress import serve
from dotenv import load_dotenv

# This is the corrected import for the chromadb-client library
import chromadb.utils.embedding_functions as embedding_functions

# Load environment variables from a .env file if it exists
load_dotenv()

# --- CONFIGURATION ---
# Reads from your .env file, but defaults to the local Docker setup
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "fF38IYJ5t11rEUYHSVuzS8aiKojPcGWNQXw0SKW4SG8")

# --- DATABASE CLIENTS ---
print("Loading Sentence Transformer model for ChromaDB embeddings...")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
print("Sentence Transformer model loaded.")

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
chroma_collection = chroma_client.get_or_create_collection(
    name="verified_reports",
    embedding_function=sentence_transformer_ef
)

# Initialize Neo4j driver
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
print("Connected to ChromaDB and local Neo4j.")

# --- FLASK APP ---
app = Flask(__name__)

# --- HELPER FUNCTIONS ---
def perform_reasoning(analysis_results):
    """Applies rules to the analysis results to get a final verdict."""
    is_claim = analysis_results.get("is_potential_claim", False)
    consistency = analysis_results.get("multimodal_consistency_score", 0.0)

    if not is_claim:
        return "NOT_A_CLAIM"
    
    if consistency > 0.6:
        return "VERIFIED_CONSISTENT_CLAIM"
    else:
        return "MISLEADING_MEDIA_CLAIM" if consistency < 0.1 else "UNVERIFIED_CLAIM"

def store_in_databases(data, verdict):
    """Stores the final results in Chroma DB and Neo4j."""
    item_id = data.get('id', 'unknown')
    
    # Handle different text fields from different platforms
    if 'text' in data and data['text']:
        document_text = data['text']
    elif 'title' in data and 'description' in data:
        document_text = f"{data.get('title', '')}: {data.get('description', '')}"
    elif 'title' in data:
        document_text = data['title']
    else:
        document_text = ""

    print(f"Storing verdict '{verdict}' for post {item_id}.")
    try:
        metadata_to_store = {
            "verdict": verdict,
            "source": data.get("source_url", "N/A"),
            "author": data.get("author", "N/A")
        }

        chroma_collection.add(
            documents=[document_text],
            metadatas=[metadata_to_store],
            ids=[item_id]
        )

        with neo4j_driver.session() as session:
            session.run("""
                MERGE (p:Post {id: $id})
                SET p.text = $text, p.verdict = $verdict, p.author = $author, p.url = $url, p.platform = $platform, p.timestamp = timestamp()
            """, id=item_id, text=document_text, verdict=verdict, author=data.get('author', 'N/A'), url=data.get('source_url', 'N/A'), platform=data.get('platform', 'unknown'))
        print(f"Successfully stored post {item_id}.")
    except Exception as e:
        print(f"Error storing data for post {item_id}: {e}")

@app.route("/reason", methods=["POST"])
def reason():
    """Receives enriched data, performs reasoning, and stores the results."""
    enriched_data = request.get_json()
    item_id = enriched_data.get("source_data", {}).get("id", "unknown")
    print(f"Reasoner received item: {item_id}")
    
    analysis_results = enriched_data.get("analysis_results", {})
    final_verdict = perform_reasoning(analysis_results)
    
    store_in_databases(enriched_data.get("source_data", {}), final_verdict)
    
    return jsonify({"status": "success", "id": item_id, "verdict": final_verdict})

if __name__ == "__main__":
    print("Reasoning Agent Service starting on port 5002...")
    serve(app, host="0.0.0.0", port=5002)

