# Install required libraries
# pip install langchain faiss-cpu openai

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Step 1: Load Documents
def load_documents():
    # Example: Load documents from text files
    loader = TextLoader("sample_docs.txt")
    documents = loader.load()
    return documents

# Step 2: Split Documents into Chunks
def split_documents(documents):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    return docs

# Step 3: Create FAISS Vector Store
def create_vector_store(docs):
    embeddings = OpenAIEmbeddings()  # Generate embeddings using OpenAI
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Step 4: Set Up RetrievalQA
def create_qa_chain(vector_store):
    retriever = vector_store.as_retriever()
    llm = OpenAI(model="text-davinci-003")  # Use OpenAI's GPT model
    qa_chain = RetrievalQA(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# Step 5: Query the System
def ask_question(qa_chain, question):
    result = qa_chain.run(question)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print("\nSource Documents:")
    for doc in result["source_documents"]:
        print(doc.metadata["source"])

# Main Workflow
if __name__ == "__main__":
    # Load and preprocess documents
    documents = load_documents()
    docs = split_documents(documents)

    # Create FAISS vector store and QA chain
    vector_store = create_vector_store(docs)
    qa_chain = create_qa_chain(vector_store)

    # Ask a question
    ask_question(qa_chain, "What is the capital of France?")
