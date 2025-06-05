import os
import openai
from langchain import hub
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv


os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

chat_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
prompt = hub.pull('rlm/rag-prompt')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
