import streamlit as st
import os
import sys
import traceback
import openai
from dotenv import load_dotenv
import torch
from typing import List, Dict, Any

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
    .main {
        padding: 2rem;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
    }
    .chat-message .content {
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

# Initialize OpenAI client
if "openai_client" not in st.session_state:
    st.session_state.openai_client = openai.OpenAI(api_key=st.secrets["CHAT_TOKEN"])

# Disable PyTorch JIT to avoid path issues
torch.jit.script = lambda x: x

# Initialize the chatbot
@st.cache_resource
def load_chatbot():
    try:
        from chatbot import initialize_chatbot
        graph = initialize_chatbot()
        if graph is None:
            st.error("Failed to initialize the chatbot. Please check the console for errors.")
            return None
        return graph
    except Exception as e:
        st.error(f"Error loading chatbot: {str(e)}")
        st.code(traceback.format_exc())
        return None

# Main app
def main():
    st.title("gprMax Documentation Chatbot")
    st.markdown("""
    This chatbot can answer questions about gprMax documentation. 
    Ask any question about gprMax and I'll try to find the answer in the documentation.
    """)
    
    # Initialize the chatbot if not already done
    if st.session_state.chatbot is None:
        with st.spinner("Initializing chatbot..."):
            graph = load_chatbot()
            if graph is not None:
                st.session_state.chatbot = graph
                st.success("Chatbot initialized successfully!")
            else:
                st.error("Failed to initialize chatbot. Please check the console for errors.")
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
                    # Initialize the state properly
                    initial_state = {
                        "question": prompt,
                        "context": [],
                        "answer": ""
                    }
                    result = st.session_state.chatbot.invoke(initial_state)
                    
                    # Handle the result based on its type
                    if isinstance(result, dict):
                        if "answer" in result:
                            response = result["answer"]
                        else:
                            response = str(result)
                    else:
                        response = str(result)
                        
                    st.markdown(response)
                except Exception as e:
                    error_message = f"Error processing query: {str(e)}"
                    st.error(error_message)
                    response = error_message
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 