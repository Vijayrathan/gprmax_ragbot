import os
import sys
import chromadb
from chatbot import (
    train_on_new_documents,
    TRAINING_DATA_DIR,
    qa_collection,
    simulation_collection,
    db_client,
    get_embeddings,
    store_embeddings,
    chunk_text
)

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
        train_on_new_documents()
        
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