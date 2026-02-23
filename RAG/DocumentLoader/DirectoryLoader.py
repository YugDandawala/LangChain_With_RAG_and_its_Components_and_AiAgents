from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
        
# It will load all the pdf files from books folder which is sequentially available to this file 
# similarly we can load all types of files and particular types of files by there extention through glob pattern
loader=DirectoryLoader(
    path='books',
    glob="*.pdf", # for all PDf
    # glob="**/*.txt"  --> #for all txt files
    # glob="data/*.csv"  --> #for all csv  files
    # glob="**/*"  --> #for all  files
    loader_cls=PyPDFLoader
)
docs=loader.load()

# <-- when there are more no of documents to load then we use load_lazy because it returns a generator rather then a list 
#      and you  load one document at a time. 

# docs=loader.lazy_load()

for document in docs:
    print(document.metadata)