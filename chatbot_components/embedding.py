import os
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime
from chatbot_components.simulation_preprocessing import parse_gprmax_input, simulation_description
from chatbot_components.component_init import chat_client

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
            
            # Parse the simulation file dynamically
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
