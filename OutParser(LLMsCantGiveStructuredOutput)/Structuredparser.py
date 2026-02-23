# *** It is removed from latest version of langchain***

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv

# It gives response according to schemas we give
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema=[
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic')
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
        template="Give 3 facts about  {topic}  \n  {format_instruction}",
        input_variables=['topic'],
        partial_variables={'format_instruction':parser.get_format_instructions()}
    )

chain=template | model | parser
result=chain.invoke({'topic':"Black hole"})
print(result)