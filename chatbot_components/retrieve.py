from chatbot_components.embedding import get_embeddings
from chatbot_components.component_init import chat_client

def retrieve(state,qa_collection,simulation_collection,reranker):
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

