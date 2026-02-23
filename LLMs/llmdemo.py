# How to load any LLM model 

from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm=OpenAI(model="gpt-3.5-turbo-instruct")
result=llm.invoke("What is the Capital of India")
# This is the Query the User will ask to this model

print(result.content)

