# This is a dynamic prompt for the list of messages if we want

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Create prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} Expert"),
    ("human", "Explain in simple terms, what is {topic}?")
])

# Initialize the model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="chat-completion"
)
model = ChatHuggingFace(llm=llm)

print("Chat started! Type 'exit' to stop.\n")

while True:
    domain=input("Enter a Domain :")
    topic = input("Enter a Topic : ")

    if topic.lower() in ["exit", "quit", "bye"]:
        print("Exiting chat...")
        break

    prompt = chat_template.invoke({"domain": domain, "topic": topic})
    response = model.invoke(prompt)

    print("\nAI:", response.content ,"\n")
