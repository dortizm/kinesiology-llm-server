"""A conversational retrieval chain."""

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOllama
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.HttpClient(settings=Settings(allow_reset=True),host='chroma', port=8000)
collection = chroma_client.get_collection(name="kinesiology") #aca va cualquier nombre que identifique el trabajo
db4 = Chroma(
    client=chroma_client,
    collection_name="kinesiology",
    embedding_function=embedding_function,
)
retriever = db4.as_retriever()

memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

model = ChatOllama(base_url='http://ollama:11434', model="mistral")

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

facilitator = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)
