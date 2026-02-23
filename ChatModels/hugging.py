from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="chat-completion",  
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Write a 4 line poem for cricket")
print(result.content)
