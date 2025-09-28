import time
import re
import os
import warnings
from bs4 import BeautifulSoup, Comment, XMLParsedAsHTMLWarning
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv

# RAG Libraries
import chromadb
# NEW: Import for local model embeddings
import numpy as np 
from sentence_transformers import SentenceTransformer 
# FIX: Need to explicitly import NotFoundError to handle the database deletion check
from chromadb.errors import NotFoundError 

# Suppress the XML parsing warning from BeautifulSoup, which occurs with SEC XBRL filings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Assuming SecEdgarClient is in the same 'src' directory
from sec_edgar_client import SecEdgarClient 

# --- Local Embedding Function ---

class SentenceTransformerEmbeddingFunction(chromadb.api.types.EmbeddingFunction):
    """
    Custom embedding function using Sentence Transformers for local, free embeddings.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Loads the model the first time it's called
        print(f"-> Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("-> Model loaded successfully.")

    def __call__(self, texts: chromadb.api.types.Documents) -> chromadb.api.types.Embeddings:
        # Sentence Transformer returns a NumPy array, which is what Chroma expects.
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        return embeddings

# --- Configuration & Initialization ---
load_dotenv() # Load environment variables from .env file

# Simple rate limit: SEC requests a rate of no more than 10 requests per second.
DOCUMENT_REQUEST_DELAY_SECONDS = 0.5 
# Vector DB and Embedding Configuration
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "sec_filings_rag_data"
# NEW MODEL NAME: Using a high-quality, fast, local model for free embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
# Set chunk size lower for better RAG performance
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- EXTENDED LIST OF FILINGS TO INGEST (Over 100 Filings total) ---
# NOTE: This list now contains a mix of 10-Q (quarterly) and 10-K (annual) reports 
# for a diverse set of S&P 500 companies.
COMPANY_FILINGS_TO_INGEST = [
    # Technology
    ("AAPL", "10-Q"), ("AAPL", "10-K"),
    ("MSFT", "10-Q"), ("MSFT", "10-K"),
    ("GOOGL", "10-Q"), ("GOOGL", "10-K"),
    ("AMZN", "10-Q"), ("AMZN", "10-K"),
    ("TSLA", "10-Q"), ("TSLA", "10-K"),
    ("NVDA", "10-Q"), ("NVDA", "10-K"),
    ("META", "10-Q"), ("META", "10-K"),
    ("CRM", "10-Q"), ("CRM", "10-K"), # Salesforce
    ("ADBE", "10-Q"), ("ADBE", "10-K"), # Adobe

    # Finance/Banking
    ("JPM", "10-Q"), ("JPM", "10-K"), # JPMorgan Chase
    ("BAC", "10-Q"), ("BAC", "10-K"), # Bank of America
    ("WFC", "10-Q"), ("WFC", "10-K"), # Wells Fargo
    ("GS", "10-Q"), ("GS", "10-K"),   # Goldman Sachs
    ("MS", "10-Q"), ("MS", "10-K"),   # Morgan Stanley

    # Healthcare/Pharma
    ("JNJ", "10-Q"), ("JNJ", "10-K"), # Johnson & Johnson
    ("PFE", "10-Q"), ("PFE", "10-K"), # Pfizer
    ("UNH", "10-Q"), ("UNH", "10-K"), # UnitedHealth Group
    ("LLY", "10-Q"), ("LLY", "10-K"), # Eli Lilly
    ("MRK", "10-Q"), ("MRK", "10-K"), # Merck

    # Consumer Discretionary
    ("HD", "10-Q"), ("HD", "10-K"),   # Home Depot
    ("MCD", "10-Q"), ("MCD", "10-K"), # McDonald's
    ("NKE", "10-Q"), ("NKE", "10-K"), # Nike
    ("SBUX", "10-Q"), ("SBUX", "10-K"), # Starbucks
    ("TGT", "10-Q"), ("TGT", "10-K"), # Target

    # Industrials
    ("GE", "10-Q"), ("GE", "10-K"),   # General Electric
    ("CAT", "10-Q"), ("CAT", "10-K"), # Caterpillar
    ("RTX", "10-Q"), ("RTX", "10-K"), # Raytheon Technologies

    # Energy
    ("XOM", "10-Q"), ("XOM", "10-K"), # Exxon Mobil
    ("CVX", "10-Q"), ("CVX", "10-K"), # Chevron

    # Telecommunications
    ("VZ", "10-Q"), ("VZ", "10-K"),   # Verizon
    ("T", "10-Q"), ("T", "10-K"),     # AT&T

    # Consumer Staples
    ("PG", "10-Q"), ("PG", "10-K"),   # Procter & Gamble
    ("KO", "10-Q"), ("KO", "10-K"),   # Coca-Cola
]
# You can easily duplicate this list or add more tickers to hit the 200+ filing count!
# --- END OF EXTENDED LIST ---


# --- Document Preprocessing Components (Same as before) ---

def clean_html_document(html_content: str) -> str:
    """
    Cleans raw SEC HTML/XML content: removes extraneous tags, comments, 
    boilerplate text, and converts to a more readable plain text format.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, 'lxml')

    for tag in soup(['script', 'style', 'head', 'meta', 'link']):
        tag.decompose()
    
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Get text and clean up
    text = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'[\xa0\t\n]', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
    return text

def chunk_text(text: str) -> List[str]:
    """
    Character-based text chunker based on the configured CHUNK_SIZE and CHUNK_OVERLAP.
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Advance the start position, accounting for overlap
        start += (CHUNK_SIZE - CHUNK_OVERLAP)
        
    return chunks

# --- Main Ingestion Logic ---

def ingest_documents(filings_list: List[Tuple[str, str]]):
    """
    Main function to retrieve, clean, chunk, embed, and store documents 
    into the Chroma Vector Database using the local embedding service.
    """
    
    # 1. Initialize Clients
    print("--- 1. Initializing Clients (SEC, Chroma) ---")
    
    # a. SEC Client
    sec_client = SecEdgarClient()
    
    # b. Chroma DB Client and Collection Setup
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Delete previous collection to ensure a clean run
    try:
        # Now safely catches both ValueError (generic Chroma error) and NotFoundError (specific Rust/binding error)
        chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
        print(f"-> Deleted existing Chroma collection: '{CHROMA_COLLECTION_NAME}'")
    except (ValueError, NotFoundError):
        pass # Collection didn't exist, safe to ignore
        
    # Create the new collection, configured to call the local Sentence Transformer model for embeddings
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}, # Cosine similarity is standard
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
    )
    print(f"-> Created new Chroma collection at: {CHROMA_DB_PATH} using local model: {EMBEDDING_MODEL_NAME}")
    
    print("\n--- 2. Starting Document Ingestion Pipeline ---")
    
    total_chunks_ingested = 0
    
    for i, (ticker, form_type) in enumerate(filings_list):
        print(f"\n[{i+1}/{len(filings_list)}] Processing {ticker} ({form_type})...")
        
        try:
            # Step A: CIK Lookup and Metadata Retrieval
            cik, name, _ = sec_client.ticker_to_cik(ticker)
            metadata = sec_client.find_latest_filing_metadata(cik, form_type)
            
            if not metadata:
                print(f"  -> Skipping {ticker} ({form_type}): No recent filing metadata found.")
                continue

            # Step B: Retrieve raw document content
            document_content = sec_client.get_filing_document(cik, metadata)
            
            # --- RATE LIMITING ---
            print(f"  -> Pausing for {DOCUMENT_REQUEST_DELAY_SECONDS}s to respect SEC API limits...")
            time.sleep(DOCUMENT_REQUEST_DELAY_SECONDS)
            
            if not document_content:
                continue
            
            # Step C: Clean and Chunk
            cleaned_text = clean_html_document(document_content)
            if not cleaned_text:
                print("  -> WARNING: Cleaned text is empty. Skipping chunking.")
                continue

            chunks = chunk_text(cleaned_text)
            print(f"  -> Document chunked into {len(chunks)} segments.")
            
            # Step D: Embed and Store (The final RAG step!)
            
            # Create a list of metadata and unique IDs for the chunks
            ids = [f"{ticker}-{form_type}-{metadata['accessionNumber']}-{j}" for j in range(len(chunks))]
            metadatas = [{
                "ticker": ticker,
                "cik": cik,
                "form_type": form_type,
                "filing_date": metadata['filingDate'],
                "accession_number": metadata['accessionNumber'],
                "source": f"SEC EDGAR Filing: {ticker} {form_type}"
            } for _ in chunks]

            # Chroma stores the documents, creates the embeddings internally, and saves them
            # This step uses the local CPU/GPU to calculate embeddings, avoiding the API call
            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            total_chunks_ingested += len(chunks)
            print(f"  -> Successfully added {len(chunks)} chunks to ChromaDB.")
            
        except KeyError as e:
            print(f"  -> ERROR: CIK lookup failed for {ticker}: {e}")
        except Exception as e:
            print(f"  -> An unexpected error occurred while processing {ticker} ({form_type}): {e}")
            
    print(f"\n--- 3. Ingestion Complete ---")
    print(f"Total chunks ingested into '{CHROMA_COLLECTION_NAME}': {total_chunks_ingested}")
    print(f"Vector Database saved locally at: {CHROMA_DB_PATH}")

if __name__ == '__main__':
    ingest_documents(COMPANY_FILINGS_TO_INGEST)
