from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_knowledge_base():
    loader = TextLoader("medical_knowledge.txt", encoding="utf-8")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    db = FAISS.from_documents(chunks, embeddings)
    return db

def get_relevant_info(query, db):
    results = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    return context