from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# loader=PyMuPDFLoader('speech.pdf')
# docs=loader.load()

text="""Nature is the intricate and beautiful web of life that surrounds us, encompassing everything from towering mountains and vast oceans to tiny insects and delicate wildflowers. It provides the resources that sustain life, including air, water, and food, and offers spaces for recreation, reflection, and inspiration. Observing natural landscapes can evoke a sense of wonder and humility, reminding humans of the complexity and interconnectedness of the world we inhabit.
Forests, rivers, and wetlands play crucial roles in maintaining ecological balance. They act as natural filters, purifying air and water, and serve as habitats for countless species of plants and animals. Seasonal cycles and weather patterns influence these ecosystems, creating a dynamic and ever-changing environment. Nature’s resilience is remarkable, yet it is also fragile and sensitive to human actions such as deforestation, pollution, and climate change, which can disrupt these delicate systems.
Beyond its ecological importance, nature has profound impacts on human well-being. Time spent outdoors can reduce stress, improve mental health, and foster creativity. Many cultures revere elements of nature in their traditions, art, and spirituality, highlighting the deep connection humans have with the natural world. Preserving and respecting nature ensures that future generations can experience its beauty and benefit from its life-sustaining resources."""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=5
)

# result=splitter.split_documents(docs)
result=splitter.split_text(text)

print(len(result))
print(result)