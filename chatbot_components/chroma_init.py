import chromadb

# Initialize ChromaDB with error handling

class FallbackChroma:
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
    



def initialize_chromadb():
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
        return qa_collection, simulation_collection, chat_history
    except Exception as e:
        print(f"Warning: ChromaDB initialization failed: {e}")
        print("Falling back to in-memory storage")
        CHROMA_AVAILABLE = False
        # Create simple in-memory storage as fallback
        qa_collection = FallbackChroma()
        simulation_collection = FallbackChroma()
        chat_history = FallbackChroma()

        return qa_collection, simulation_collection, chat_history