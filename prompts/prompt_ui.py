from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="chat-completion"
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

paper_input=st.selectbox("Select Research Paper Name",["Select...","Attention is all you need",
                        "Bert:pre-training of deep bidirectional transformer","GPT-3:language models are fee -shot learners",
                        "Diffusion Models  beat GANs on image synthesis"])

style_input=st.selectbox("Select Explanation style",["Beginner-friendly","Technical","Code-oriented","Mathematical"])
length_input=st.selectbox("Select Explantion length",["Short(1-2 paragraph)","Medium(3-5 paragraphs)","Long(detailed explanation)"])

template=load_prompt("template.json")


if st.button('Summarize'):
    # Chain is made
    chain=template|model
    result=chain.invoke({'paper_input':paper_input,
                    'style_input':style_input,
                    'length_input':length_input})
    st.write(result.content)


