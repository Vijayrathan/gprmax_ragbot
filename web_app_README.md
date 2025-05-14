# GPRMax Assistant Web Application

A modern, responsive web interface for the GPRMax Chatbot, providing an intuitive way to interact with the GPRMax documentation and get answers about Ground Penetrating Radar simulations.

## Features

- **Beautiful Chat Interface**: Modern UI with animations and visual feedback
- **Real-time Responses**: Get answers from the GPRMax documentation in real-time
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices
- **Typing Indicators**: Visual cues when the assistant is generating a response

## Technologies Used

- **Backend**: Flask, OpenAI, LangChain, ChromaDB
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with responsive design
- **Icons**: Font Awesome
- **Fonts**: Google Fonts (Roboto)

## Installation

1. Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

2. Set up your environment variables (the chatbot.py file should already be configured)

3. Run the Flask application:

```bash
python app.py
```

4. Open your browser and navigate to `http://127.0.0.1:5001`

## Usage

1. Type your question about GPRMax or Ground Penetrating Radar simulations in the input field
2. Press Enter or click the send button to submit your question
3. The assistant will process your query and provide a response based on the GPRMax documentation

## Development

- **Static files**: CSS and JavaScript files are in the `static` directory
- **Templates**: HTML templates are in the `templates` directory
- **Backend logic**: The main application is in `app.py` which interfaces with `chatbot.py`

## License

This project is part of the GPRMax software package documentation and research tools.

## Acknowledgements

- GPRMax Documentation
- OpenAI for their powerful language models
- The LangChain project for their RAG implementation
