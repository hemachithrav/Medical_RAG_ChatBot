from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from src.prompt import *
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY



embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot" 

from langchain_pinecone import PineconeVectorStore
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# Create LLM and create a chain

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

system_prompt = (
    "You are a medical assistant for question-answering tasks. "
    "Use ONLY the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "This is not medical advice.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)




#Initialize Flask
app=Flask(__name__)

#Default route
@app.route("/")
def index():
    return render_template('chat.html')

#Route once user sends a query
@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]   # ✅ matches AJAX
        print("User:", msg)

        # 👉 Greeting handling
        if msg.lower() in ["hi", "hello", "hey"]:
            return "👋 Hi! I'm MediAssist. How can I help you today?"

        # 👉 RAG call
        response = rag_chain.invoke({"input": msg})

        answer = response["answer"]
        print("Response:", answer)

        return answer   # ⚠️ returning string (not JSON)

    except Exception as e:
        print("Error:", e)
        return "❌ Something went wrong"


if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)




