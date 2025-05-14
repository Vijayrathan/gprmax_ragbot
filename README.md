# gprMax Documentation Chatbot

A Streamlit-based chatbot that answers questions about gprMax documentation using RAG (Retrieval-Augmented Generation).

## Features

- Interactive web interface built with Streamlit
- RAG-based question answering using gprMax documentation
- Persistent vector database for efficient retrieval
- Reranking of retrieved documents for better relevance
- Fallback mechanism for environments where ChromaDB is not available

## Setup

### Option 1: Using the setup script (recommended)

1. Clone this repository:

   ```
   git clone <repository-url>
   cd gprmax_rag
   ```

2. Run the setup script:

   ```
   python setup.py
   ```

3. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

### Option 2: Manual setup

1. Clone this repository:

   ```
   git clone <repository-url>
   cd gprmax_rag
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Make sure you have the gprMax documentation PDF file (`docs-gprmax-com-en-latest.pdf`) in the root directory.

4. Run the Streamlit app:

   ```
   streamlit run streamlit_app.py
   ```

5. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501).

## Usage

1. Once the app is running, you'll see a chat interface where you can ask questions about gprMax.
2. Type your question in the input field and press Enter.
3. The chatbot will retrieve relevant information from the documentation and generate an answer.
4. The chat history will be maintained during your session.

## How It Works

1. The app first checks if the vector database is populated. If not, it processes the PDF documentation.
2. When you ask a question, the app:
   - Generates embeddings for your query
   - Retrieves relevant document chunks from the vector database
   - Reranks the retrieved chunks for better relevance
   - Uses the retrieved context to generate a comprehensive answer

## Fallback Mechanism

The chatbot includes a fallback mechanism for environments where ChromaDB is not available or encounters issues:

- If ChromaDB initialization fails, the app will automatically switch to an in-memory storage solution
- If the PDF file is not found, the app will use dummy data for testing
- These fallbacks ensure the app can still run and demonstrate functionality even in limited environments

## Troubleshooting

- If you encounter any issues with the OpenAI API, make sure your API key is correctly set in the code.
- If the chatbot fails to initialize, check that the PDF file exists and is readable.
- For any other issues, check the console output for error messages.
- If you see ChromaDB-related errors, the app should automatically fall back to in-memory storage.


