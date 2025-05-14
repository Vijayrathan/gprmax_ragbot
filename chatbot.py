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
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Remove or comment out this line
# embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize components
prompt = hub.pull('rlm/rag-prompt')

# Initialize OpenAI client with error handling
try:
    chat_client = openai.OpenAI(
        api_key="",
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
    # Check if collection exists
    try:
        collection = db_client.get_collection(name="gprMax_docs")
        print("Using existing collection")
    except:
        collection = db_client.create_collection(name="gprMax_docs")
        print("Created new collection")
    
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
    # Create a simple in-memory storage as fallback
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
    
    collection = SimpleStorage()
    chat_history = SimpleStorage()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    session_id: str  # Add session_id to track conversations

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

def store_embeddings(chunks, embeddings, sources):
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
    # Get embedding for the query using OpenAI
    query_embedding = get_embeddings([query])[0]
    
    # Query the database - reduce number of results
    results = collection.query(query_embeddings=[query_embedding], n_results=10)
    
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
    
    # Get recent chat history
    history = get_chat_history(state["session_id"])
    
    # Build messages with history
    messages = [
        {"role": "system", "content": """You are a helpful AI assistant.
        You are specialized in GPRMAX software.
        You are speaking in a friendly and professional manner.
        You should not answer questions that are not related to GPRMAX software.
        Your primary goal is to help user with simulation related questions.
        If user asks for simulation examples or anything related to simulation, you should provide all the parameters of input file (.in file)
        If user asks for simulation examples or anything related to simulation, you always format the file in a code block, separated from the rest of the answer"""}
    ]
    
    # Add chat history
    for item in history:
        messages.append({"role": "user", "content": item["question"]})
        messages.append({"role": "assistant", "content": item["answer"]})
    
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

# Function to initialize the chatbot
def initialize_chatbot():
    
    # Check if we need to load and process the PDF
    if collection.count() == 0:
        print("No documents in collection. Loading PDF...")
        try:
            pdf_loader = PyPDFLoader("docs-gprmax-com-en-latest.pdf")
            pages_data = pdf_loader.load()
            
            # Add additional metadata to each page
            for i, page in enumerate(pages_data):
                page.metadata.update({
                    "source": "docs-gprmax-com-en-latest.pdf",
                    "page_number": i + 1,
                    "total_pages": len(pages_data),
                    "document_type": "gprMax Documentation"
                })
            
            # Process each page and create meaningful chunks
            pages_chunks = []
            for page in pages_data:
                chunks = chunk_text(page.page_content, chunk_size=200)  # Larger chunks ##PDF
                if chunks:
                    for chunk in chunks:
                        if len(chunk) > 50:  # Only keep meaningful chunks
                            # Create a new Document for each chunk with the page's metadata
                            chunk_doc = { 'page_content':chunk, 'metadata':page.metadata.copy() }
                            pages_chunks.append(chunk_doc)
                else:
                    print(f"No valid chunks for page: {page.metadata.get('page_number', '')}")
            
            print(f"Created {len(pages_chunks)} chunks from {len(pages_data)} pages")
            
            # Exit if no valid chunks
            if not pages_chunks:
                print("Error: No valid chunks to process. Check your content extraction.")
                return None
            
            # Get the chunks and sources
            chunk_texts = [item["page_content"] for item in pages_chunks]
            sources = [item["metadata"]["source"] for item in pages_chunks]
            
            # Process in smaller batches
            print("Generating embeddings with OpenAI ada-002...")
            batch_size = 5  # Much smaller batch size
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
                else:
                    print(f"Failed to get embeddings for batch {i//batch_size + 1}")
            
            # Final check and save
            if all_embeddings:
                print(f"Successfully embedded {len(all_embeddings)} chunks out of {len(chunk_texts)}")
                store_embeddings(chunks=valid_chunks, embeddings=all_embeddings, sources=valid_sources)
            else:
                print("Failed to generate any valid embeddings.")
                return None
        except Exception as e:
            print(f"Error processing PDF: {e}")
            # Create some dummy data for testing
            
    
    # Build and compile the graph - using a string instead of START
    graph_builder = StateGraph(State).add_sequence([retrieve, generate_response])
    graph_builder.add_edge("__start__", "retrieve")  # Use string "__start__" instead of START
    graph = graph_builder.compile()
    
    return graph

# Function to process a query
def process_query(graph, query, session_id="default"):
    try:
        result = graph.invoke({
            "question": query,
            "session_id": session_id
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


    
        
    
        

        
    
    


