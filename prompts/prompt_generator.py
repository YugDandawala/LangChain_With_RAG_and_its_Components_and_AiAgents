# This is a dynamic prompt for the single messages if we want

from langchain_core.prompts import PromptTemplate

template=PromptTemplate(template="""Please summarize the reaserch paper titled {paper_input} with the following specification
Explanation style :{style_input} and Explanation length :{length_input} wioth 1.mathematical detailed 2.analogies
Ensure the summary is clear ,accurate and aligned with the provide style and length""",
# validate true means if any parameter is missing it will tell you 
input_variables=['paper_input','style_input','length_input'],validate_template=True
)

template.save("template.json")