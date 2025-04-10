import re
# from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer,models, CrossEncoder
import chromadb
from langchain import hub
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
import openai
from langgraph.graph import START, StateGraph
import os
from langchain_community.document_loaders import PyPDFLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Remove or comment out this line
# embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

db_client = chromadb.PersistentClient(path="./chroma_db")
# Check if collection exists
try:
    collection = db_client.get_collection(name="gprMax_docs")
    print("Using existing collection")
except:
    collection = db_client.create_collection(name="gprMax_docs")
    print("Created new collection")
prompt=hub.pull('rlm/rag-prompt')
chat_client = openai.OpenAI(api_key="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")

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

# def get_subpage_links(url, base_url="https://docs.gprmax.com/en/latest/", visited=None, max_depth=3, current_depth=0):
#     if visited is None:
#         visited = set()
#     if url in visited or current_depth >= max_depth:
#         return []
#     visited.add(url)
    
#     try:
#         headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
        
#         # Check if content is HTML before parsing
#         if 'text/html' in response.headers.get('Content-Type', ''):
#             soup = BeautifulSoup(response.text, 'html5lib')
            
#             links = []

#             for a in soup.find_all('a', href=True):
#                 absolute_link = urljoin(url, a['href'])
#                 # Filter for internal links under the docs base URL
#                 if absolute_link.startswith(base_url):
#                     links.append(absolute_link)
#                     # Recursively fetch links from the subpage
#                     links.extend(get_subpage_links(absolute_link, base_url, visited, max_depth, current_depth + 1))
            
#             return list(set(links))
#         else:
#             print(f"Skipping non-HTML content at {url}")
#             return []
#     except Exception as e:
#         print(f"Error fetching {url}: {e}")
#         return []

# def clean_text(raw_text: str) -> str:

#     text = raw_text.strip()
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'Page \d+ of \d+', '', text)  # remove pagination if present
#     text = re.sub(r'©\s.*', '', text)            # remove copyright notices
    
#     return text

# def extract_content_from_page():
#     main_url = "https://docs.gprmax.com/en/latest/"
#     all_links = get_subpage_links(main_url)
#     print("Found {} pages.".format(len(all_links)))
#     pages_data=[]
#     for link in all_links:
#         try:
#             headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
#             resp = requests.get(link, headers=headers, timeout=10)
#             soup = BeautifulSoup(resp.text, 'html5lib')
            
#             # Focus on the main content area if possible
#             main_content = soup.find('div', class_='document') or soup.find('article') or soup
            
#             # Get text but filter out navigation and other non-content areas
#             text_content = ""
#             if main_content:
#                 # Skip navigation and sidebar elements
#                 for nav in main_content.find_all(['nav', 'aside']):
#                     nav.decompose()
                
#                 # Get the remaining text
#                 text_content = clean_text(main_content.get_text(separator=" "))
            
#             # Only add pages with substantial content
#             if len(text_content) > 200:  # At least 200 characters to be meaningful
#                 title = soup.title.string if soup.title else soup.h1.string if soup.h1 else ""
#                 pages_data.append({'url': link, 'title': title, 'text': text_content})
#                 print(f"Added page: {title} - {len(text_content)} chars")
#             else:
#                 print(f"Skipping page with insufficient content: {link}")
#         except Exception as e:
#             print(f"Error extracting content from {link}: {e}")
    
#     return pages_data


def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()

    print(len(words))
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        print(i,i+chunk_size)
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
    if not query_embedding:
        return {"context": ["Error: Could not generate embeddings for query"]}
    
    # Query the database
    results = collection.query(query_embeddings=[query_embedding], n_results=20)
    
    # Rerank results
    pairs = [[query, results["metadatas"][0][i]["text"]] for i in range(len(results["metadatas"][0]))]
    scores = reranker.predict(pairs)
    
    # Sort by reranker scores
    reranked_results = sorted(zip(results["metadatas"][0], scores), key=lambda x: x[1], reverse=True)
    
    # Limit context size by truncating or using fewer documents
    context_texts = []
    total_tokens = 0
    max_tokens = 12000  # Conservative limit to leave room for completion
    
    for item in reranked_results:
        text = item[0]["text"]
        # Rough token estimate (words / 0.75)
        estimated_tokens = len(text.split()) / 0.75
        if total_tokens + estimated_tokens > max_tokens:
            break
        context_texts.append(text)
        total_tokens += estimated_tokens
    
    return {"context": context_texts}

def generate_response(state):
    query = prompt.invoke({"question": state["question"], "context": state["context"]})

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": f'Context:\n{state["context"]}\n\nQuestion: {query}'}
    ]
    response = chat_client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return {"answer": response.choices[0].message.content}


if __name__=='__main__':
    # Test the embedding endpoint first
    print("Testing OpenAI embeddings endpoint...")
    test_result = get_embeddings(["This is a test of the OpenAI embeddings API"])
    if not test_result:
        print("Error: Initial OpenAI embeddings test failed. Please check your API key and connection.")
        exit(1)
    else:
        print("OpenAI embeddings test successful!")
    
    # Continue with the rest of your code
    # pages_data = extract_content_from_page() ##SCRAPING
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
        exit(1)
    
    # Get the chunks and sources
    chunk_texts = [item["page_content"] for item in pages_chunks]
    sources = [item["metadata"]["source"] for item in pages_chunks]
    
    # # Process in smaller batches
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
    
    # # Final check and save
    if all_embeddings:
        print(f"Successfully embedded {len(all_embeddings)} chunks out of {len(chunk_texts)}")
        collection = store_embeddings(chunks=valid_chunks, embeddings=all_embeddings, sources=valid_sources)
    else:
        print("Failed to generate any valid embeddings. Exiting.")
        exit(1)

    graph_builder = StateGraph(State).add_sequence([retrieve, generate_response])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    while(1):
        user_question = input('Enter your query: ')
        if user_question.lower() == "exit":
            break
        try:
            result = graph.invoke({"question": user_question})
            print(f'Answer: {result["answer"]}')
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            print("Please try a more specific or shorter query.")


    
        
    
        

        
    
    


