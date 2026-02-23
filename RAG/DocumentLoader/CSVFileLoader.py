from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

loader=CSVLoader(file_path='titanic.csv')
docs=loader.load()

print(docs[0])

# There are lots of Loaders watch out on langchain site there is loader avaible for all things when needed for projects
# We can also create our own loaders if we doesnt have particular loader