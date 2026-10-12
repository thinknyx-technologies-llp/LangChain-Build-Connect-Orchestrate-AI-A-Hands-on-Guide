from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

loader = PyPDFLoader("text_mining.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150,
    separators = ["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(docs)

print("Total documents: ", len(docs))
print("total chunks created: ", len(chunks))

embeddings = GoogleGenerativeAIEmbeddings(model= "models/gemini-embedding-001")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",
    collection_name="text_mining"
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 4}
)

prompt = ChatPromptTemplate.from_template(
    """
Answer the question based oonly on the following context.
If the answer isn't in the context, simply say don'y know.

Context: 
{context}

Question: {question}
"""
)

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

def format_docs(docs):

    print("Retreived Context")

    for i, doc in enumerate(docs, 1):
        print(f"\n Context Chunk {i}")
        print(doc.page_content)

    return "\n\n".join(
        doc.page_content for doc in docs
    )

format_docs_runnable = RunnableLambda(format_docs)

def debug_prompt(prompt_value):
    print("PROMPT SENT TO LLM")
    print(prompt_value.to_string())
    return prompt_value

debug_prompt_runnable = RunnableLambda(debug_prompt)

rag_chain = (
    {
        "context": retriever | format_docs_runnable,
        "question": RunnablePassthrough()
    }
    | prompt
    | debug_prompt_runnable
    | model
    | StrOutputParser()
)

query = input("Question: ")

answer = rag_chain.invoke(query)
print("Final Answer", answer)

