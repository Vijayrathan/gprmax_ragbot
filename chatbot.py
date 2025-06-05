from chatbot_components.retrieve import retrieve
from chatbot_components.generator import generate_response
from chatbot_components.chroma_init import initialize_chromadb
from chatbot_components.component_init import prompt,reranker
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

#Initialize the chatbot state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    session_id: str
    mode: str  # 'qa' for general Q&A, 'simulation' for simulation file generation

# initialize the chatbot
def initialize_chatbot():
    qa_collection, simulation_collection, chat_history = initialize_chromadb()
    
    # Create wrapper functions for the workflow nodes
    def retrieve_node(state):
        return retrieve(state, qa_collection, simulation_collection, reranker)
    
    def generate_node(state):
        return generate_response(state, prompt, chat_history)
    
    # Build and compile the graph
    graph_builder = StateGraph(State).add_sequence([retrieve_node, generate_node])
    graph_builder.add_edge("__start__", "retrieve_node")
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


    
        
    
        

        
    
    


