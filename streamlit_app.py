import streamlit as st
import os
import traceback
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="gprMax Documentation Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main { padding: 2rem; }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-message.user { background-color: #e6f3ff; }
    .chat-message.assistant { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

# Initialize the chatbot
@st.cache_resource
def initialize_chatbot():
    try:
        # Delay importing torch until needed
        from chatbot import initialize_chatbot
        graph = initialize_chatbot()
        return graph
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        st.code(traceback.format_exc())
        return None

# Main app
def main():
    st.title("gprMax Documentation Chatbot")
    st.markdown("Ask any question about gprMax and I'll try to find the answer in the documentation.")
    
    # Initialize chatbot if not already done
    if st.session_state.chatbot is None:
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = initialize_chatbot()
            if st.session_state.chatbot:
                st.success("Chatbot initialized successfully!")
            else:
                st.error("Failed to initialize chatbot.")
                return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about gprMax..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Simple input state for the graph
                    result = st.session_state.chatbot.invoke({
                        "question": prompt,
                        "context": [],
                        "answer": ""
                    })
                    
                    # Extract answer from result
                    if isinstance(result, dict) and "answer" in result:
                        response = result["answer"]
                    else:
                        response = str(result)
                    
                    st.markdown(response)
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    response = error_message
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()