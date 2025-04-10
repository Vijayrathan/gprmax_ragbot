import streamlit as st
import os
import sys
from chatbot import initialize_chatbot, process_query

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

# Initialize the chatbot
@st.cache_resource
def load_chatbot():
    return initialize_chatbot()

# Main app
def main():
    st.title("gprMax Documentation Chatbot")
    st.markdown("""
    This chatbot can answer questions about gprMax documentation. 
    Ask any question about gprMax and I'll try to find the answer in the documentation.
    """)
    
    # Initialize the chatbot
    chatbot = load_chatbot()
    
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
                response = process_query(chatbot, prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 