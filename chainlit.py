import os
from dotenv import load_dotenv, find_dotenv
import chainlit as cl
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv(find_dotenv())

# Hugging Face setup
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")

repo_id = "google/flan-t5-base"  # You can change this to your preferred model
llm = HuggingFaceEndpoint(
    repo_id=repo_id, 
    temperature=0.5,
    max_length=512,
    huggingfacehub_api_token=hf_api_token
)

# Qdrant setup
qdrant_url = os.getenv("QDRANT_URL", "").split("#")[0].strip()  # Remove any comments and whitespace
if not qdrant_url:
    raise ValueError("QDRANT_URL not found in environment variables. Please set it to the correct URL of your Qdrant server.")

qdrant_collection_name = "chatbot_collection"

try:
    client = QdrantClient(url=qdrant_url)
    # Test the connection
    client.get_collections()
except Exception as e:
    raise ValueError(f"Failed to connect to Qdrant at {qdrant_url}. Error: {str(e)}")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

# Global variable to store the vector store
vector_store = None

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome! Please upload a PDF, TXT, or CSV file to begin.").send()

    files = None
    while not files:
        files = await cl.AskFileMessage(
            content="Please upload a PDF, TXT, or CSV file to begin.",
            accept=["application/pdf", "text/plain", "text/csv"],
            max_size_mb=20,
            timeout=180,
        ).send()

    if files:
        file = files[0]
        await process_file(file)

async def process_file(file):
    global vector_store

    try:
        await cl.Message(content=f"Processing file: {file.name}").send()

        # Load and process the document based on file type
        if file.type == "application/pdf":
            loader = PyPDFLoader(file.path)
        elif file.type == "text/plain":
            loader = TextLoader(file.path)
        elif file.type == "text/csv":
            loader = CSVLoader(file.path)
        else:
            raise ValueError(f"Unsupported file type: {file.type}")

        documents = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Create and store the embeddings using Qdrant
        vector_store = Qdrant.from_documents(
            documents=splits,
            embedding=embeddings,
            url=qdrant_url,
            collection_name=qdrant_collection_name,
        )

        # Create the conversational chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            vector_store.as_retriever(),
            return_source_documents=True
        )

        cl.user_session.set("qa_chain", qa_chain)

        await cl.Message(content=f"File '{file.name}' has been processed. You can now ask questions about its content.").send()
    except Exception as e:
        await cl.Message(content=f"An error occurred while processing the file: {str(e)}").send()
        await cl.Message(content="Please try uploading the file again or contact support if the issue persists.").send()

@cl.on_message
async def main(message: cl.Message):
    qa_chain = cl.user_session.get("qa_chain")
    if qa_chain:
        try:
            result = qa_chain({"question": message.content, "chat_history": []})
            answer = result["answer"]
            await cl.Message(content=answer).send()
        except Exception as e:
            await cl.Message(content=f"An error occurred while processing your question: {str(e)}").send()
            await cl.Message(content="Please try asking your question again or upload a new file if the issue persists.").send()
    else:
        await cl.Message(content="Please upload a file before asking questions.").send()