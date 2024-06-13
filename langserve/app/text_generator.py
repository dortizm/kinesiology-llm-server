from langchain.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = ChatOllama(base_url='http://ollama:11434', model="mistral")
prompt = """Text Generator.
    You are a virtual assistant in the area of kinesiology.
    You must generate short texts of no more than 100 words.
    The generated text must be only in Spanish language.
    Return a result in this json schema: "generated:text, explanation:text"
    The generated text must be associated with the following context: {context}"""

prompt = ChatPromptTemplate.from_template(prompt)

text_generator = (prompt | model | StrOutputParser())