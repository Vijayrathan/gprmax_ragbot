from flask import Flask, render_template, request, jsonify, session
import chatbot
import uuid

app = Flask(__name__)
app.secret_key = 'x12300' 

chat_graph = None

@app.before_request
def initialize_chatbot():
    global chat_graph
    if chat_graph is None:
        print("Initializing chatbot...")
        chat_graph = chatbot.initialize_chatbot()
        print("Chatbot initialization complete!")
    
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    if 'mode' not in session:
        session['mode'] = 'qa'  # Default to QA mode

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
        # Check if this is a mode switch command
        if user_question.lower().startswith("/mode"):
            mode = user_question.lower().split()[1] if len(user_question.split()) > 1 else "qa"
            if mode not in ["qa", "simulation"]:
                return jsonify({'error': "Invalid mode. Please use '/mode qa' or '/mode simulation'"})
            session['mode'] = mode
            return jsonify({'answer': f"Switched to {mode} mode. {'Ask any questions about GPR and GPRMax.' if mode == 'qa' else 'I will help you create simulation files. What kind of simulation would you like to create?'}"})
        
        # Process the query with the current mode
        current_mode = session.get('mode', 'qa')
        answer = chatbot.process_query(chat_graph, user_question, session['session_id'], current_mode)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 