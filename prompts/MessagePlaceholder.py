# If our conversation is stored in database and user comes and ask about some last conversation but our llm will not be able to help for that conversation
# so we load that conversation so our llm wil respond according to last conversation

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# Initialize the model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="chat-completion"
)
model = ChatHuggingFace(llm=llm)
chat_history=[]

with open("chat_history.txt") as f:
    for line in f:
        line = line.strip()
        if line.startswith("human:"):
            chat_history.append(HumanMessage(content=line.replace("human:", "").strip()))
        elif line.startswith("ai:"):
            chat_history.append(AIMessage(content=line.replace("ai:", "").strip()))
        
prompt=chat_template.invoke({'chat_history':chat_history,'query':'when will i get my refund'})
response = model.invoke(prompt)

print(response.content)
