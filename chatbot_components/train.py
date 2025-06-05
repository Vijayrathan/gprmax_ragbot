import os
import sys
import chromadb
from embedding import (
    get_embeddings,
    store_embeddings,
    process_document
)
import hashlib
import json
from chatbot_components.component_init import chat_client
TRAINING_DATA_DIR = "../training_data"
TRAINING_STATE_FILE = "../training_state.json"


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

def train_on_new_documents(qa_collection,simulation_collection):
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
def initialize_chromadb():
    """Initialize ChromaDB with proper error handling"""
    try:
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
        
        return db_client, qa_collection, simulation_collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        sys.exit(1)

def main():
    print("Starting GPRMax Assistant training process...")
    print("-----------------------------------")
    
    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    db_client, qa_collection, simulation_collection = initialize_chromadb()
    
    # Check if training data directory exists
    if not os.path.exists(TRAINING_DATA_DIR):
        print(f"Error: Training data directory '{TRAINING_DATA_DIR}' not found!")
        print(f"Please create the directory and add your training files.")
        sys.exit(1)
    
    # Check if there are any files in the training directory
    training_files = []
    for root, _, files in os.walk(TRAINING_DATA_DIR):
        for file in files:
            if file.endswith(('.pdf', '.in')):
                training_files.append(os.path.join(root, file))
    
    if not training_files:
        print(f"Error: No training files found in '{TRAINING_DATA_DIR}'!")
        print("Please add .pdf or .in files to the training directory.")
        sys.exit(1)
    
    # Count files by type
    pdf_files = [f for f in training_files if f.endswith('.pdf')]
    in_files = [f for f in training_files if f.endswith('.in')]
    
    print(f"Found {len(training_files)} training files:")
    print(f"- {len(pdf_files)} PDF files for QA mode")
    print(f"- {len(in_files)} .in files for Simulation mode")
    for file in training_files:
        print(f"- {os.path.basename(file)}")
    
    print("\nStarting training process...")
    try:
        # Train on the documents
        train_on_new_documents(qa_collection,simulation_collection)
        
        # Print training summary
        qa_count = qa_collection.count()
        sim_count = simulation_collection.count()
        total_documents = qa_count + sim_count
        
        print("\nTraining completed successfully!")
        print(f"Total documents in database: {total_documents}")
        print(f"- QA documents: {qa_count}")
        print(f"- Simulation documents: {sim_count}")
        
        if total_documents == 0:
            print("\nWarning: No documents were added to the database!")
            print("This might indicate a problem with the training process.")
            print("Please check the logs above for any errors.")
            sys.exit(1)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 