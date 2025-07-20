from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load your PDF
loader = PyPDFLoader("smart_fridge.pdf")  # Replace with your actual PDF file name
pages = loader.load()

# Step 2: Split the text into small chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

# Step 3: Print the number of chunks
print(f"Total chunks created: {len(chunks)}")

# Optional: print the first chunk
print(chunks[0].page_content)
