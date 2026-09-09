from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("Sample.pdf")

documents = loader.load()
print(type(documents))
print("TOTAL DOCUMENTS: ", len(documents))

for i, doc in enumerate(documents):
    print(f"PAGE {i+1}")
    print(doc.page_content)
    print(doc.metadata)
    print("\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 30
)

chunks = splitter.split_documents(documents)

print("TOTAL CHUNKS: ", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"CHUNK {i+1}")
    print(chunk.page_content)
    print("METADATA: ", chunk.metadata)
    print("\n")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = FAISS.from_documents(
    documents = chunks,
    embedding = embeddings
)

print("Vector Store Created Successfully")

retriever = vector_store.as_retriever()
question = input("Question: ")
retrieved_docs = retriever.invoke(question)

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

for i, doc in enumerate(retrieved_docs):
    print(f"\nDocument {i+1}\n")
    print(doc.page_content)

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

prompt = PromptTemplate.from_template(
    """
Answer the question only using the context below.
Context: {context}

Question: {question}
"""
)

response = model.invoke(prompt.invoke(
    {
        "context" : context,
        "question": question
    }
))

print(response.content[0]["text"])
