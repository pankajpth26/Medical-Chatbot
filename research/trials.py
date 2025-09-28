from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

#Load PDF data
def load_pdf(data_dir):
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    return documents

extracted_data = load_pdf("data/")
print(f"Extracted {len(extracted_data)} pages from PDF.")

# Step 2: Split text into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = text_split(extracted_data)
print(f"Split into {len(text_chunks)} chunks.")

# Step 3: Download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embeddings = download_hugging_face_embeddings()

# Test embeddings
query_result = embeddings.embed_query("Hello world")
print(f"Embedding length: {len(query_result)}")

# Step 4: Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

index_name = "medical-chatbot"  # Ensure this index is created in Pinecone dashboard

# Step 5: Create embeddings and store in Pinecone
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
print("Embeddings stored in Pinecone.")

# Step 6: Query the vector store (example)
query = "What is the treatment for fever?"
docs = docsearch.similarity_search(query, k=3)
print(docs)

# Additional imports for full chain (from langchain)
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Initialize OpenAI LLM
llm = OpenAI(model_name="text-davinci-003", openai_api_key=os.environ.get('OPENAI_API_KEY'))

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=docsearch.as_retriever()
)

# Run a query through the chain
result = qa_chain({"query": query})
print(result["result"])
