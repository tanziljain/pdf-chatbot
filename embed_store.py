from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load and split the PDF
loader = PyPDFLoader("smart_fridge.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

# Step 2: Create embeddings from chunks
embeddings = HuggingFaceEmbeddings()

# Step 3: Store them in a FAISS vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 4: Save the FAISS index to local disk
vectorstore.save_local("faiss_index")

print("✅ Embeddings created and stored successfully!")