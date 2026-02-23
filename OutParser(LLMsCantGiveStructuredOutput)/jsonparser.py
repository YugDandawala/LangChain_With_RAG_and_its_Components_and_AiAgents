from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
parser=JsonOutputParser()

template=PromptTemplate(
        # template="Give me a name, age and city of a fictional person{name} \n {format_instruction}",
        
        # --> if we do this it will give you a json output only according to it but we want some specific type of json we use structured output parser
        template="Give me a name, age and city of a fictional person \n {format_instruction}",
        # input_variables=['name],
        input_variables=[],
        partial_variables={'format_instruction':parser.get_format_instructions()}
    )
chain=template | model | parser
result=chain.invoke({})

print(result)

# chain=template1 | model | parser | template2 | model | parser
# result=chain.invoke({"topic":"Black hole"})

# print(result)