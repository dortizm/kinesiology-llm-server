import chromadb
from chromadb.config import Settings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain import hub
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from pprint import pprint
import re

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def retrieve(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    # Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(str(d))
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    Transform the query to produce a better question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    Args:
        state (dict): The current graph state
    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    filtered_documents = state["documents"]
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    Args:
        state (dict): The current graph state
    Returns:
        str: Decision for next node to call
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]
    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def translate_query(state):
    """
    Generar respuesta
    Args:
        state (dict): El estado actual del grafo
    Returns:
        state (dict): Nueva clave añadida al estado, generación, que contiene la generación del LLM
    """
    print("---TRADUCCION---")
    generation = state["generation"]
    context = state["documents"]
    #context = "The following is a conversation with an artificial intelligence research assistant. The assistant's responses should be easy to understand even for elementary school students and should not exceed 3 lines. Only do what the question asks."
    # Generación RAG
    #question= "translate this English text into Spanish: " + question
    translate_generation = translate_chain.invoke({"question": generation})
    return {"generation": translate_generation}

def formal_query(state):
    """
    Generar respuesta
    Args:
        state (dict): El estado actual del grafo
    Returns:
        state (dict): Nueva clave añadida al estado, generación, que contiene la generación del LLM
    """
    print("---FORMAL---")
    generation = state["generation"]
    context = state["documents"]
    #context = "The following is a conversation with an artificial intelligence research assistant. The assistant's responses should be easy to understand even for elementary school students and should not exceed 3 lines. Only do what the question asks."
    # Generación RAG
    #question= "translate this English text into Spanish: " + question
    formal_generation = formal_chain.invoke({"question": generation})
    return {"generation": formal_generation}
    
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.HttpClient(settings=Settings(allow_reset=True),host='chroma', port=8000)
langchainChroma = Chroma(client = chroma_client, persist_directory="./chroma_db", collection_name = 'kinesiology_plus', embedding_function=embedding_function)
retriever = langchainChroma.as_retriever()
llm = ChatOllama(base_url='http://ollama:11434', model="mistral")

# Retrieval Grader
prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()

# Generate
#_prompt = hub.pull("rlm/rag-prompt")
mistra_prompt = hub.pull("rlm/rag-prompt-mistral")

g_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.

    Do not include any references, notes, explanations, or preambles. Provide the answer in English only.
    
    Question: {question} 
    Context: {context} 
    Answer: """,
    input_variables=["question", "context"],
)

rag_chain = mistra_prompt | llm | StrOutputParser()

# Hallucination Grader
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()

# Answer Grader
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()

# Question Re-writer
re_write_prompt = PromptTemplate(
    template="""You are a question re-writer that converts an input question to a better version that is optimized 
    for vectorstore retrieval. Look at the initial question and formulate an improved question.
    
    Do not include any citations, references, or preambles.
    
    Here is the initial question:
    -------
    {question}
    -------
    Improved question: """,
    input_variables=["question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()


# Translate Query

# Function to clean up the response
def clean_translation(response):
    # Remove any text that appears after a known marker, like "Translation:"
    print("_____________clean_translation_____________")
    print("_____________response_____________")
    cleaned_response = re.split(r'(Translation:|Here\'s the translated text|In Spanish°)', response, 1)[0]
    print("_____________cleaned_response_____________")
    return cleaned_response.strip()


t_prompt = PromptTemplate(
    template="""You are a translator converting medical text from English to Spanish. The translation should be clear and precise, without including citations or sources from the documents, and accessible for patients who are not healthcare professionals.

    Translate the following text into Spanish. Do not include any notes, explanations, references, symbols, numbers, sources, or English text.

    Here is the text in English:
    -------
    {question}
    -------
    Provide the translated text in Spanish, keeping the original format.""",
    input_variables=["question"],
)

translate_chain = t_prompt | llm | StrOutputParser() | clean_translation



# Define darle formato al texto



formal_prompt = PromptTemplate(
    template="""You are a healthcare professional who converts poorly written Spanish medical text into clear and patient-friendly Spanish. The explanation should be precise, using comprehensible terminology and an explanatory and approachable tone suitable for patients.

    Transform the following text into a clear and explanatory style, removing any sources, notes, explanations, references, symbols, numbers, and English text. If any condition related to the knee joint is mentioned, always use "artrosis de rodilla".

    Here is the poorly written Spanish text:
    -------
    {question}
    -------
    Provide the transformed explanation in clear and patient-friendly Spanish.""",
    input_variables=["question"],
)




formal_chain = formal_prompt | llm | StrOutputParser()




# Define el flujo de trabajo
workflow = StateGraph(GraphState)

# Define los nodos
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("translate_query", translate_query)  # translate_query
workflow.add_node("formal_query", formal_query)  # formal_query


# Construir el grafo
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": "translate_query",
        "not useful": "transform_query",
    },
)
workflow.add_edge("translate_query", "formal_query")
workflow.set_finish_point("formal_query")

# Compilar
bot = workflow.compile()