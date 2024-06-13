from langchain.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = ChatOllama(base_url='http://ollama:11434', model="mistral")
prompt = """What is the text analysis of this user response? 
    Respond with a value between 0 and 1, where 0 is negative and 1 is affirmative.
    The user's response is in Spanish.
    Return a result in this json schema: "value:int, explanation:text"
    This is the user response: {user_response}"""

prompt = ChatPromptTemplate.from_template(prompt)

text_analysis = (prompt | model | StrOutputParser())