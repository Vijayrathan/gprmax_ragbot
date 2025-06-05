from chatbot_components.embedding import get_embeddings
from datetime import datetime
def store_chat_history(chat_history,session_id: str, question: str, answer: str):
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

def get_chat_history(chat_history,session_id: str, limit: int = 5):
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
