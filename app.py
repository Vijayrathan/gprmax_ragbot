from flask import Flask, render_template, request, jsonify, session
import chatbot
import uuid

app = Flask(__name__)
app.secret_key = ''  # Required for session management

# Global variable to store the chat graph
chat_graph = None

@app.before_request
def initialize_chatbot():
    global chat_graph
    if chat_graph is None:
        print("Initializing chatbot...")
        chat_graph = chatbot.initialize_chatbot()
        print("Chatbot initialization complete!")
    
    # Initialize session if it doesn't exist
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    user_question = data.get('question', '')
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        answer = chatbot.process_query(chat_graph, user_question, session['session_id'])
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 