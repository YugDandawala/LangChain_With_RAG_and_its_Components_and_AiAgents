from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
    
prompt=PromptTemplate(
        template="Answer the following question \n {question} from the following text \n{text}",
        input_variables=['question','text'],
 )

parser=StrOutputParser()

url="https://www.flipkart.com/samsung-galaxy-a55-5g-awesome-navy-128-gb/p/itm7ac5d2771f7a0?pid=MOBGYT2H4PGHBRHJ&lid=LSTMOBGYT2H4PGHBRHJTLRZ3Y&marketplace=FLIPKART&cmpid=content_mobile_8965229628_gmc"

# You can also send list of Url's here
loader=WebBaseLoader(url)
docs=loader.load()

chain = prompt | model | parser
result=chain.invoke({'question':'what is the product','text':docs[0].page_content})
print(result)