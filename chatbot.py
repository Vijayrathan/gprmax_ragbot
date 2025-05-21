import re
# from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer,models, CrossEncoder
import os
from langchain import hub
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
import openai
from langgraph.graph import StateGraph
# Remove the START import and use a string instead
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from dotenv import load_dotenv
from datetime import datetime
import hashlib
import json
import dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
TRAINING_DATA_DIR = "training_data"
TRAINING_STATE_FILE = "training_state.json"

def parse_gprmax_input(file_path):
    """Parse a GPRMax input file and extract key parameters"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    params = {
        "title": "",
        "domain": "",
        "resolution": "",
        "time_window": "",
        "material": "",
        "waveform": "",
        "source": "",
        "receiver": "",
        "source_steps": "",
        "receiver_steps": "",
        "geometry": []
    }

    for line in lines:
        if line.startswith("#title:"):
            params["title"] = line.strip().split(":")[1].strip()
        elif line.startswith("#domain:"):
            params["domain"] = line.strip().split(":")[1].strip()
        elif line.startswith("#dx_dy_dz:"):
            params["resolution"] = line.strip().split(":")[1].strip()
        elif line.startswith("#time_window:"):
            params["time_window"] = line.strip().split(":")[1].strip()
        elif line.startswith("#material:"):
            params["material"] = line.strip().split(":")[1].strip()
        elif line.startswith("#waveform:"):
            params["waveform"] = line.strip().split(":")[1].strip()
        elif line.startswith("#hertzian_dipole:"):
            params["source"] = line.strip().split(":")[1].strip()
        elif line.startswith("#rx:"):
            params["receiver"] = line.strip().split(":")[1].strip()
        elif line.startswith("#src_steps:"):
            params["source_steps"] = line.strip().split(":")[1].strip()
        elif line.startswith("#rx_steps:"):
            params["receiver_steps"] = line.strip().split(":")[1].strip()
        elif line.startswith("#cylinder:") or line.startswith("#box:"):
            params["geometry"].append(line.strip())

    return params

def simulation_description(params):
    """Generate a descriptive text from simulation parameters"""
    return (
        f"Simulation titled '{params['title']}' with domain {params['domain']} m, "
        f"resolution {params['resolution']} m, and time window {params['time_window']} s. "
        f"Material defined as {params['material']}. "
        f"Waveform used: {params['waveform']}. "
        f"Source at {params['source']}, receiver at {params['receiver']}. "
        f"Source steps: {params['source_steps']}, receiver steps: {params['receiver_steps']}. "
        f"Geometry includes: {'; '.join(params['geometry'])}."
    )

# Initialize components
prompt = hub.pull('rlm/rag-prompt')

# Initialize OpenAI client with error handling
try:
    chat_client = openai.OpenAI(
        api_key=dotenv.getenv("OPENAI_API_KEY")
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    chat_client = None

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Initialize ChromaDB with error handling
try:
    import chromadb
    # Initialize with persistent storage
    db_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create separate collections for QA and Simulation modes
    try:
        qa_collection = db_client.get_collection(name="gprmax_qa_docs")
        print("Using existing QA collection")
    except:
        qa_collection = db_client.create_collection(name="gprmax_qa_docs")
        print("Created new QA collection")
    
    try:
        simulation_collection = db_client.get_collection(name="gprmax_simulations")
        print("Using existing Simulation collection")
    except:
        simulation_collection = db_client.create_collection(name="gprmax_simulations")
        print("Created new Simulation collection")
    
    # For backward compatibility with existing code
    collection = qa_collection
    
    # Create a separate collection for chat history
    try:
        chat_history = db_client.get_collection(name="chat_history")
        print("Using existing chat history collection")
    except:
        chat_history = db_client.create_collection(name="chat_history")
        print("Created new chat history collection")
    
    CHROMA_AVAILABLE = True
except Exception as e:
    print(f"Warning: ChromaDB initialization failed: {e}")
    print("Falling back to in-memory storage")
    CHROMA_AVAILABLE = False
    # Create simple in-memory storage as fallback
    class SimpleStorage:
        def __init__(self):
            self.data = []
            self.embeddings = []
            self.metadata = []
        
        def add(self, embeddings, metadatas, ids):
            self.embeddings.extend(embeddings)
            self.metadata.extend(metadatas)
            self.data.extend(ids)
        
        def query(self, query_embeddings, n_results=10):
            # Simple cosine similarity search
            results = []
            for i, query_embedding in enumerate(query_embeddings):
                similarities = []
                for j, embedding in enumerate(self.embeddings):
                    # Simple dot product as similarity
                    similarity = sum(a * b for a, b in zip(query_embedding, embedding))
                    similarities.append((j, similarity))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in similarities[:n_results]]
                
                results.append({
                    "ids": [self.data[idx] for idx in top_indices],
                    "metadatas": [[self.metadata[idx]] for idx in top_indices],
                    "distances": [1 - sim for _, sim in similarities[:n_results]]
                })
            
            return results
        
        def count(self):
            return len(self.data)
    
    qa_collection = SimpleStorage()
    simulation_collection = SimpleStorage()
    chat_history = SimpleStorage()
    
    # For backward compatibility with existing code
    collection = qa_collection

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    session_id: str
    mode: str  # 'qa' for general Q&A, 'simulation' for simulation file generation

# System prompts for different modes
QA_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in GPRMAX software.
You are speaking in a friendly and professional manner.
You should not answer questions that are not related to GPRMAX software.
Your primary goal is to help users with GPR and GPRMax related questions.
Focus on providing clear, accurate information about parameters, concepts, and general queries.
Do not generate simulation files in this mode."""

SIMULATION_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in GPRMAX software.
You are speaking in a friendly and professional manner.
Your only goal is to generate GPRMax simulation files.
When users request a simulation, ask follow-up questions if you don't have all the necessary parameters in the context:
Once you have all required information, generate a complete .in file.
Format the simulation file in a code block with the following structure:
```gprmax
# filename: simulation.in
# Your simulation data here
```
Always separate the code block from the rest of the answer with a blank line before and after."""

def store_chat_history(session_id: str, question: str, answer: str):
    """Store chat history in ChromaDB"""
    try:
        chat_history.add(
            embeddings=[get_embeddings([f"{question} {answer}"])[0]],
            metadatas=[{
                "session_id": session_id,
                "question": question,
                "answer": answer,
                "timestamp": str(datetime.now())
            }],
            ids=[f"{session_id}_{hash(f'{question}{answer}')}"]
        )
    except Exception as e:
        print(f"Error storing chat history: {e}")

def get_chat_history(session_id: str, limit: int = 5):
    """Retrieve recent chat history for a session"""
    try:
        results = chat_history.query(
            query_embeddings=[get_embeddings(["recent conversation"])[0]],
            n_results=limit
        )
        # Filter results by session_id
        history = []
        for metadata in results["metadatas"][0]:
            if metadata["session_id"] == session_id:
                history.append({
                    "question": metadata["question"],
                    "answer": metadata["answer"]
                })
        return history
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()

    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i+chunk_size]
        if len(chunk) > 50:  # Only keep chunks with at least 50 words
            chunks.append(" ".join(chunk))
    return chunks

def store_embeddings(chunks, embeddings, sources, collection):
    # Check that we have equal numbers of chunks, embeddings, and sources
    if not (len(chunks) == len(embeddings) == len(sources)):
        print(f"Warning: Mismatched lengths - chunks:{len(chunks)}, embeddings:{len(embeddings)}, sources:{len(sources)}")
        # Use the minimum length to avoid index errors
        min_length = min(len(chunks), len(embeddings), len(sources))
        chunks = chunks[:min_length]
        embeddings = embeddings[:min_length]
        sources = sources[:min_length]
    
    # Add error handling for each item
    for i, (chunk, embedding, source) in enumerate(zip(chunks, embeddings, sources)):
        try:
            if embedding is not None:
                collection.add(
                    embeddings=[embedding],  # OpenAI embeddings are already lists, no need for tolist()
                    metadatas=[{"text": chunk, "source": source}],
                    ids=[f"{source}_{hash(chunk)}"]
                )
        except Exception as e:
            print(f"Error storing embedding {i}: {e}")
    
    return collection

# Improve the get_embeddings function with better debugging
def get_embeddings(texts, model="text-embedding-ada-002"):
    if not texts:
        print("Empty texts list passed to get_embeddings")
        return []
    
    # Debug text properties
    print(f"Processing {len(texts)} texts")
    print(f"First text sample: '{texts[0][:50]}...' (length: {len(texts[0]) if texts[0] else 0})")
    
    # OpenAI recommends replacing newlines with spaces for best results
    cleaned_texts = []
    for i, text in enumerate(texts):
        if text and isinstance(text, str) and len(text.strip()) > 0:
            # Clean the text
            cleaned = text.replace("\n", " ").strip()
            # Handle extremely long texts (OpenAI has token limits)
            if len(cleaned) > 8000:  # rough character limit
                cleaned = cleaned[:8000]
            if cleaned:  # Only add non-empty strings
                cleaned_texts.append(cleaned)
        else:
            print(f"Text {i} is invalid: {type(text)}, empty: {not text}")
    
    if not cleaned_texts:
        print("No valid text to embed after cleaning")
        print(f"Original text count: {len(texts)}")
        if texts and isinstance(texts[0], str):
            print(f"First original text: '{texts[0][:100]}'")
        return []
    
    print(f"Sending {len(cleaned_texts)} texts for embedding")
    
    try:
        # Format exactly as in OpenAI docs
        response = chat_client.embeddings.create(
            model=model,
            input=cleaned_texts
        )
        print(f"Successfully got embeddings for {len(response.data)} texts")
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        return []

def retrieve(state):
    query = state["question"]
    mode = state.get("mode", "qa")
    
    print(f"Retrieving in mode: {mode}")
    
    # Get embedding for the query using OpenAI
    query_embedding = get_embeddings([query])[0]
    
    # Select appropriate collection based on mode
    target_collection = simulation_collection if mode == "simulation" else qa_collection
    print(f"Using collection: {'simulation' if mode == 'simulation' else 'qa'}")
    
    # Query the database - reduce number of results
    results = target_collection.query(query_embeddings=[query_embedding], n_results=10)
    
    # Rerank results
    pairs = [[query, results["metadatas"][0][i]["text"]] for i in range(len(results["metadatas"][0]))]
    scores = reranker.predict(pairs)
    
    # Sort by reranker scores
    reranked_results = sorted(zip(results["metadatas"][0], scores), key=lambda x: x[1], reverse=True)
    
    # Limit context size by truncating or using fewer documents
    context_texts = []
    total_tokens = 0
    max_tokens = 4000  # Reduced from 12000 to leave more room for the response
    
    for item in reranked_results:
        text = item[0]["text"]
        # More accurate token estimation (roughly 4 chars per token)
        estimated_tokens = len(text) / 4
        if total_tokens + estimated_tokens > max_tokens:
            break
        context_texts.append(text)
        total_tokens += estimated_tokens
    
    # If we have too much context, take only the top 3 most relevant chunks
    if len(context_texts) > 3:
        context_texts = context_texts[:3]
    
    return {"context": context_texts}

def generate_response(state):
    query = prompt.invoke({"question": state["question"], "context": state["context"]})
    mode = state.get("mode", "qa")
    
    print(f"Generating response in mode: {mode}")
    
    # Get recent chat history
    history = get_chat_history(state["session_id"])
    
    # Select system prompt based on mode
    system_prompt = QA_SYSTEM_PROMPT if mode == "qa" else SIMULATION_SYSTEM_PROMPT
    
    # Build messages with history
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add chat history
    for item in history:
        messages.append({"role": "user", "content": item["question"]})
        messages.append({"role": "assistant", "content": item["answer"]})
    
    # Debug context
    print(f"Mode: {mode}, Context sources:")
    for i, ctx in enumerate(state["context"]):
        if isinstance(ctx, str) and len(ctx) > 100:
            print(f"  Context {i+1}: {ctx[:100]}...")
        else:
            print(f"  Context {i+1}: {ctx}")
    
    # Add current context and question
    messages.append({"role": "user", "content": f'Context:\n{state["context"]}\n\nQuestion: {query}'})
    
    try:
        response = chat_client.chat.completions.create(
            model="gpt-4.1",
            messages=messages
        )
        # Safely access the response content
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                answer = response.choices[0].message.content
                # Store in chat history
                store_chat_history(state["session_id"], state["question"], answer)
                return {"answer": answer}
        return {"answer": "Sorry, I couldn't generate a response. Please try again."}
    except Exception as e:
        return {"answer": f"Error generating response: {str(e)}"}

def get_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_training_state():
    """Load the training state from JSON file"""
    if os.path.exists(TRAINING_STATE_FILE):
        with open(TRAINING_STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_training_state(state):
    """Save the training state to JSON file"""
    with open(TRAINING_STATE_FILE, 'w') as f:
        json.dump(state, f)

def process_document(file_path, mode="qa"):
    """Process a single document and return its chunks"""
    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    
    try:
        if mode == "qa":
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            else:
                print(f"Unsupported file type for QA mode: {file_extension}")
                return []
            
            pages_data = loader.load()
            print(f"Processing QA document: {file_name}")
            
            # Add additional metadata to each page
            for i, page in enumerate(pages_data):
                page.metadata.update({
                    "source": file_name,
                    "page_number": i + 1,
                    "total_pages": len(pages_data),
                    "document_type": "gprMax Documentation",
                    "file_type": file_extension[1:],
                    "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
            
            # Process each page and create meaningful chunks
            pages_chunks = []
            for page in pages_data:
                chunks = chunk_text(page.page_content, chunk_size=500)
                if chunks:
                    for chunk in chunks:
                        if len(chunk) > 50:  # Only keep meaningful chunks
                            chunk_doc = {'page_content': chunk, 'metadata': page.metadata.copy()}
                            pages_chunks.append(chunk_doc)
                else:
                    print(f"No valid chunks for page: {page.metadata.get('page_number', '')}")
            
            return pages_chunks
            
        elif mode == "simulation":
            if file_extension != '.in':
                print(f"Unsupported file type for Simulation mode: {file_extension}")
                return []
            
            print(f"Processing Simulation file: {file_name}")
            
            # Parse the simulation file
            params = parse_gprmax_input(file_path)
            description = simulation_description(params)
            
            # Create a single chunk with the simulation description
            chunk_doc = {
                'page_content': description,
                'metadata': {
                    "source": file_name,
                    "document_type": "gprMax Simulation",
                    "file_type": "in",
                    "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    "simulation_params": params
                }
            }
            
            return [chunk_doc]
        
        else:
            print(f"Invalid mode: {mode}")
            return []
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

def train_on_new_documents():
    """Train on new documents in the training data directory"""
    if not os.path.exists(TRAINING_DATA_DIR):
        os.makedirs(TRAINING_DATA_DIR)
        print(f"Created training data directory: {TRAINING_DATA_DIR}")
        return
    
    # Check if ChromaDB is new (empty collections)
    is_new_chromadb = qa_collection.count() == 0 and simulation_collection.count() == 0
    if is_new_chromadb:
        print("\nNew ChromaDB detected. Will train on all files regardless of state.")
    
    # Load current training state
    training_state = load_training_state()
    
    # Get all files in training data directory
    training_files = []
    for root, _, files in os.walk(TRAINING_DATA_DIR):
        for file in files:
            if file.endswith(('.pdf', '.in')):
                training_files.append(os.path.join(root, file))
    
    if not training_files:
        print("No training files found in the training data directory.")
        return
    
    # Process each file
    for file_path in training_files:
        file_hash = get_file_hash(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Determine mode based on file extension
        mode = "simulation" if file_extension == '.in' else "qa"
        
        # Process file if it's new, modified, or if ChromaDB is new
        if is_new_chromadb or file_path not in training_state or training_state[file_path] != file_hash:
            print(f"\nProcessing file: {file_path}")
            if is_new_chromadb:
                print("(Processing due to new ChromaDB)")
            elif file_path not in training_state:
                print("(Processing new file)")
            else:
                print("(Processing modified file)")
            
            # Process the document
            chunks = process_document(file_path, mode)
            
            if chunks:
                print(f"Created {len(chunks)} chunks from {file_path}")
                
                # Get the chunks and sources
                chunk_texts = [item["page_content"] for item in chunks]
                sources = [item["metadata"]["source"] for item in chunks]
                
                # Process in smaller batches
                print(f"Generating embeddings for {len(chunk_texts)} chunks...")
                batch_size = 5
                all_embeddings = []
                valid_chunks = []
                valid_sources = []
                
                for i in range(0, len(chunk_texts), batch_size):
                    batch = chunk_texts[i:i+batch_size]
                    batch_sources = sources[i:i+batch_size]
                    print(f"Processing batch {i//batch_size + 1}/{(len(chunk_texts) + batch_size - 1)//batch_size}")
                    
                    batch_embeddings = get_embeddings(batch)
                    if batch_embeddings:
                        all_embeddings.extend(batch_embeddings)
                        valid_chunks.extend(batch)
                        valid_sources.extend(batch_sources)
                        print(f"Successfully embedded batch {i//batch_size + 1}")
                    else:
                        print(f"Failed to get embeddings for batch {i//batch_size + 1}")
                
                # Save embeddings to appropriate collection
                if all_embeddings:
                    print(f"Storing {len(all_embeddings)} embeddings in {mode} collection...")
                    target_collection = simulation_collection if mode == "simulation" else qa_collection
                    store_embeddings(chunks=valid_chunks, embeddings=all_embeddings, sources=valid_sources, collection=target_collection)
                    # Update training state
                    training_state[file_path] = file_hash
                    print(f"Successfully processed {file_path}")
                else:
                    print(f"Failed to generate embeddings for {file_path}")
            else:
                print(f"No valid chunks found in {file_path}")
        else:
            print(f"Skipping unchanged file: {file_path}")
    
    # Save updated training state
    save_training_state(training_state)
    print("\nTraining state saved.")

# Function to initialize the chatbot
def initialize_chatbot():
    # Build and compile the graph
    graph_builder = StateGraph(State).add_sequence([retrieve, generate_response])
    graph_builder.add_edge("__start__", "retrieve")
    graph = graph_builder.compile()
    
    return graph

# Function to process a query
def process_query(graph, query, session_id="default", mode="qa"):
    try:
        # Check if this is a mode switch command
        if query.lower().startswith("/mode"):
            new_mode = query.lower().split()[1] if len(query.split()) > 1 else "qa"
            if new_mode not in ["qa", "simulation"]:
                return "Invalid mode. Please use '/mode qa' or '/mode simulation'"
            return f"Switched to {new_mode} mode. {'Ask any questions about GPR and GPRMax.' if new_mode == 'qa' else 'I will help you create simulation files. What kind of simulation would you like to create?'}"
        
        # Process the query with the current mode
        result = graph.invoke({
            "question": query,
            "session_id": session_id,
            "mode": mode
        })
        
        # Safely access the answer from the result
        if isinstance(result, dict) and "answer" in result:
            return result["answer"]
        elif isinstance(result, str):
            return result
        else:
            return f"Unexpected result format: {str(result)}"
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Main function for command-line usage
# if __name__=='__main__':
#     # Initialize the chatbot
#     graph = initialize_chatbot()
#     if not graph:
#         print("Failed to initialize chatbot. Exiting.")
#         exit(1)
    
#     # Command-line interface
#     print("Chatbot initialized. Type 'exit' to quit.")
#     while True:
#         user_question = input('Enter your query: ')
#         if user_question.lower() == "exit":
#             break
#         try:
#             result = process_query(graph, user_question)
#             print(f'Answer: {result}')
#         except Exception as e:
#             print(f"Error processing query: {str(e)}")
#             print("Please try a more specific or shorter query.")


    
        
    
        

        
    
    


