import chromadb
import os
import json
import requests
import warnings
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer 
from flask import Flask, request, jsonify

# --- CONFIGURATION (Must match ingestion.py) ---
CHROMA_DB_PATH = "./chroma_db" 
CHROMA_COLLECTION_NAME = "sec_filings_rag_data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
TOP_K_RESULTS = 5

# --- GEMINI CONFIGURATION ---
# The API key must be provided as an environment variable in your cloud deployment service.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_MODEL_NAME = "gemini-2.5-flash" 

# --- Global Variables for RAG components (initialized on startup) ---
app = Flask(__name__)
RAG_COLLECTION = None

# --- Local Embedding Function (Needed by ChromaDB) ---
class SentenceTransformerEmbeddingFunction(chromadb.api.types.EmbeddingFunction):
    """Custom embedding function using Sentence Transformers."""
    def __init__(self, model_name: str):
        warnings.filterwarnings("ignore")
        print(f"-> Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
    def __call__(self, texts: chromadb.api.types.Documents) -> chromadb.api.types.Embeddings:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

# --- RAG Core Logic Functions ---

def build_system_prompt() -> str:
    """Defines the LLM's role."""
    return (
        "You are an expert financial analyst. Your task is to answer the user's question "
        "concisely and accurately, based ONLY on the provided context, which consists of "
        "SEC filing excerpts. If the context does not contain the answer, state clearly "
        "that the information is not available in the provided documents. "
        "Always cite the source ticker (e.g., AAPL or MSFT) for the information you provide."
    )

def build_prompt_with_context(query: str, retrieved_context: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Combines query with retrieved context chunks."""
    context_text = "\n\n--- CONTEXT CHUNKS ---\n"
    source_tickers = []
    
    for i, item in enumerate(retrieved_context):
        ticker = item['metadata']['ticker']
        form_type = item['metadata']['form_type']
        distance = item.get('distance', 'N/A')
        
        # Add the context to the text sent to the LLM
        context_text += f"\n[SOURCE {i+1} | {ticker} {form_type} | Distance: {distance:.4f}]: {item['document']}\n"
        source_tickers.append(ticker)
    
    # The full prompt tells Gemini its role and provides the context.
    full_prompt = (
        f"USER QUERY: {query}\n"
        f"{context_text}\n"
        "--------------------------\n"
        "Based on the CONTEXT CHUNKS above, provide your expert analysis and answer the USER QUERY."
    )
    return full_prompt, list(set(source_tickers))

def call_gemini_api(full_prompt: str, system_prompt: str) -> str:
    """Calls the Gemini API to generate the final RAG answer."""
    
    if not GEMINI_API_KEY:
        return "ERROR: GEMINI_API_KEY environment variable is not set. Cannot call Gemini API."

    headers = {'Content-Type': 'application/json'}
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{ "parts": [{ "text": full_prompt }] }],
        "config": {
            "systemInstruction": system_prompt,
            "temperature": 0.1 
        }
    }
    
    print(f"-> Sending request to Gemini API ({GEMINI_MODEL_NAME})...")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status() 

        result = response.json()
        
        # Extract the generated text from the Gemini response structure
        generated_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No response text.')
        
        return generated_text
        
    except requests.exceptions.RequestException as e:
        # Check for 400/403 errors which often indicate an invalid API key
        status_code = response.status_code if 'response' in locals() else 'N/A'
        return f"GEMINI API COMMUNICATION ERROR: Status {status_code}. Details: {e}. Check API key validity and billing status."
    except Exception as e:
        return f"GEMINI RESPONSE ERROR: Could not parse response. Details: {e}"

# --- Flask App Initialization and Endpoint ---

@app.before_first_request
def initialize_rag():
    """Initializes the ChromaDB connection once when the application starts."""
    global RAG_COLLECTION
    print("--- RAG API Service Initializing (Gemini LLM) ---")
    
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"FATAL ERROR: Database path not found at {CHROMA_DB_PATH}. Cannot start RAG service.")
        RAG_COLLECTION = None
        return

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        RAG_COLLECTION = chroma_client.get_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME
            )
        )
        print(f"✅ Database connection successful. DB Size: {RAG_COLLECTION.count()} chunks.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load ChromaDB collection. Error: {e}")
        RAG_COLLECTION = None

@app.route('/query', methods=['POST'])
def query_rag():
    """The main RAG endpoint for external queries."""
    if RAG_COLLECTION is None:
        return jsonify({"error": "RAG service failed to initialize. Database missing or corrupted."}), 500

    try:
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({"error": "Missing 'query' parameter in request body."}), 400

        print(f"\n--- New Query Received: {query} ---")

        # 1. Retrieval
        results = RAG_COLLECTION.query(
            query_texts=[query],
            n_results=TOP_K_RESULTS,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results or not results['documents'] or not results['documents'][0]:
            return jsonify({"answer": "Retrieval failed: No relevant documents found.", "sources": []})

        retrieved_data = []
        for i in range(len(results['documents'][0])):
            retrieved_data.append({
                'document': results['documents'][0][i], 
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'source_id': f"{results['metadatas'][0][i]['ticker']} {results['metadatas'][0][i]['form_type']}"
            })

        # 2. Generation (LLM Call Prep)
        system_prompt = build_system_prompt()
        full_prompt, source_tickers = build_prompt_with_context(query, retrieved_data)

        # 3. LLM Call (Gemini)
        final_answer = call_gemini_api(full_prompt, system_prompt)

        return jsonify({
            "query": query,
            "answer": final_answer,
            "sources_used": list(set(source_tickers)),
            "retrieved_chunks_count": len(retrieved_data)
        })

    except Exception as e:
        print(f"RAG processing failed: {e}")
        return jsonify({"error": f"Internal server error during RAG processing: {e}"}), 500

if __name__ == '__main__':
    # This block is for local testing of the API server only
    initialize_rag()
    app.run(host='0.0.0.0', port=5000)
