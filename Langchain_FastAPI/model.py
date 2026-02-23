from fastapi import FastAPI, Request, Form 
from fastapi.templating import Jinja2Templates 
from fastapi.responses import HTMLResponse 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
from langchain_postgres import PGVector 
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage 
from dotenv import load_dotenv 
from langchain_core.documents import Document 
import pandas as pd
import markdown
load_dotenv() 

user = "postgres" 
password = "Admin1234" 
host = "localhost" 
database = "Podcast"
collection = "embedded_vectors" 
connectDB = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{database}"

llm_endpoint = HuggingFaceEndpoint( 
    repo_id="meta-llama/Meta-Llama-3-70B-Instruct", 
    task="text-generation", 
    temperature=0.1 
) 

model = ChatHuggingFace(llm=llm_endpoint) 

app = FastAPI() 

templates = Jinja2Templates(directory="templates") 

df = pd.read_csv("Flipkart_Data.csv")

texts = [] 
for idx, row in df.iterrows(): 
    formatted = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    texts.append(formatted) 

documents = [Document(page_content=t) for t in texts] 
    
embeddings = HuggingFaceEndpointEmbeddings( 
    model="sentence-transformers/all-MiniLM-L6-v2"
) 

db = PGVector.from_documents( 
    documents=documents, 
    embedding=embeddings, 
    connection=connectDB, 
    collection_name=collection 
) 

SYSTEM_PROMPT = """ You are a Mobile Phone Seller. All your responses must be in **Markdown format**. **Rules:** 
1. **Phone Queries:** - If the user asks about phones, reply with **full specifications** and **useful details**. - Present the information in **Markdown tables or lists** for clarity.
2. **Unrelated Queries:** - If the user asks anything unrelated to mobile phones, reply exactly: > This is not my scope, I don't answer things like this. - Do **not** give any recommendations or extra information. 
3. **Generalized Requests:** - If the conversation is about a phone and the user asks for generalizations (e.g., "What phones was I searching before?"), search the chat history and provide **only relevant phone information** in Markdown format. - If the request is unrelated to the current phone discussion, reply exactly: > This is not my scope, I don't answer things like this. 
4. **Strict Compliance:** - Follow all rules very strictly. - Any violation may result in the AI being blocked or disabled permanently. **Markdown Guidelines:** - Use tables, headings, or lists to display phone specs. - Example for a single phone: - If user ask about anything related to phones reply in this format even if ask for list of phones or list of anything reply in this md file format - It is not like you should answer by full specfications of phones but you strictly need to give output in md file format - If user ask give list of phones with highest Ram present you should provide output in given md file example
markdown
### Samsung Galaxy S21

### Motorola Phones

Here are the Motorola phones:

#### List of Phones

### Google Pixel Phones
Here are the Google Pixel phones:

| P_name                | Color      | Storage | RAM  | Ratings | Display | Battery | Price     |
|---------------------- |------------|---------|------|---------|---------|---------|-----------|
| Google Pixel 10 Pro   | Moonstone  | 256 GB  | 16 GB| 279     | 16.0 cm | 4870 mAh| ₹1,09,999 |
| Google Pixel 10 Pro   | Obsidian   | 256 GB  | 16 GB| 279     | 16.0 cm | 4870 mAh| ₹1,09,999 |
| Google Pixel 9 Pro    | Snow       | 128 GB  | 12 GB| 1,017   | 15.9 cm | 5124 mAh| ₹71,999   |
| Google Pixel 9 Pro    | Snow       | 256 GB  | 12 GB| 1,017   | 15.9 cm | 5124 mAh| ₹76,999   |
| Google Pixel 9 Pro    | Snow       | 512 GB  | 12 GB| 1,017   | 15.9 cm | 5124 mAh| ₹91,999   |
| Google Pixel 9        | Lemongrass | 128 GB  | 8 GB | 101     | 15.7 cm | 4320 mAh| ₹49,999   |
| Google Pixel 9        | Lemongrass | 256 GB  | 8 GB | 101     | 15.7 cm | 4320 mAh| ₹59,999   |
| Google Pixel 9 Pro    | Obsidian   | 128 GB  | 12 GB| 1,017   | 16.2 cm | 5124 mAh| ₹71,999   |
| Google Pixel 9 Pro    | Obsidian   | 256 GB  | 12 GB| 1,017   | 16.2 cm | 5124 mAh| ₹76,999   |
| Google Pixel 9 Pro    | Obsidian   | 512 GB  | 12 GB| 1,017   | 16.2 cm | 5124 mAh| ₹91,999   |

"""

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, output: str = ""):
    md_html = markdown.markdown(output, extensions=['tables', 'fenced_code'])
    return templates.TemplateResponse("index.html", {"request": request, "Output": output, "md_html": md_html})

# Global in-memory store
chat_history = []

@app.post("/generate")
async def generate(request: Request, response: str = Form(...)):
    global chat_history

    retrieved_docs = db.similarity_search(response,k=40)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    chat_history.append(HumanMessage(content=response))

    messages = [
    SystemMessage(content=SYSTEM_PROMPT),*chat_history,
    HumanMessage(content=f"Context:\n{context}\n\nAnswer the question: {response}")
    ]

    answer = model.invoke(messages).content
    md_html = markdown.markdown(answer, extensions=['tables', 'fenced_code', 'toc'])
    chat_history.append(AIMessage(content=answer))

    return templates.TemplateResponse(
        "index.html",
        {"request": request,"Output": answer, "md_html": md_html}
    )
