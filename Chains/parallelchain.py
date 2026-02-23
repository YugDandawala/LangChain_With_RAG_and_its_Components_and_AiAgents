from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
        template="Generate a short and simple notes from the following \n{text}",
        input_variables=['topic']
    )
prompt2=PromptTemplate(
        template="Generative a 5 question and answer from the following text \n{text}",
        input_variables=['text']
    )

prompt3=PromptTemplate(
        template="Merge the provided notes and quiz into a single document \n notes  ->{notes} and quiz -> {quiz}",
        input_variables=['notes','quiz']
    )

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 | model2 | parser
})

merge_chain=prompt3 | model1| parser

chain=parallel_chain | merge_chain

text="""Linear Regression in Machine learning
Last Updated : 14 Oct, 2025
Linear regression is a type of supervised machine-learning algorithm that learns from the labelled datasets and maps the data points with most optimized linear functions which can be used for prediction on new datasets. It assumes that there is a linear relationship between the input and output, meaning the output changes at a constant rate as the input changes. This relationship is represented by a straight line.

For example we want to predict a student's exam score based on how many hours they studied. We observe that as students study more hours, their scores go up. In the example of predicting exam scores based on hours studied. Here

Independent variable (input): Hours studied because it's the factor we control or observe.
Dependent variable (output): Exam score because it depends on how many hours were studied.
We use the independent variable to predict the dependent variable.
Why Linear Regression is Important?
Here’s why linear regression is important:

Simplicity and Interpretability: It’s easy to understand and interpret, making it a starting point for learning about machine learning.
Predictive Ability: Helps predict future outcomes based on past data, making it useful in various fields like finance, healthcare and marketing.
Basis for Other Models: Many advanced algorithms, like logistic regression or neural networks, build on the concepts of linear regression.
Efficiency: It’s computationally efficient and works well for problems with a linear relationship.
Widely Used: It’s one of the most widely used techniques in both statistics and machine learning for regression tasks.
Analysis: It provides insights into relationships between variables (e.g., how much one variable influences another)."""

result=chain.invoke({'text':text})

print(result)

# To visually see the graph of chain
chain.get_graph().print_ascii()
