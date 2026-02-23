# PyPDFLoader is for textual pdf
# PDFPlumberLoader is used when pdf have more tabular data 
# UnstructuredPDFLoader or AmazonTextractLoader is used when we have Scanned/images  PDFs
# PyMuPDFLoader is used when we Need layout and image data
# UnstructuredPDFLoader is used when we want best structure extraction
# More on langchain site
from langchain_community.document_loaders import PyPDFLoader
        
loader=PyPDFLoader('speech.pdf')
docs=loader.load()

print(docs[0].page_content)