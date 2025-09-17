from langchain_groq import ChatGroq
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os 

# Load environment variables
load_dotenv()


# Initialize Azure OpenAI
AZURE_ENDPOINT = os.getenv("AZURE_PHI4_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_PHI4_API")
GROQ_API = os.getenv("GROQ_API")


# llm = ChatOpenAI(
#     base_url=AZURE_ENDPOINT,
#     api_key=AZURE_API_KEY,
#     model="phi-4",
#     temperature=0.7
# )

llm1 = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API
)