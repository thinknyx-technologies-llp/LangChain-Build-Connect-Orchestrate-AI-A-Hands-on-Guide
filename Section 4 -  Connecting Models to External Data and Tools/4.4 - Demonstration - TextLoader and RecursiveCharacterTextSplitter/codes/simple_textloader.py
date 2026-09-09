from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("sample.txt")

documents = loader.load()

print(type(documents))
print(len(documents))
# print(documents)
print(documents[0].page_content)
print(documents[0].metadata)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 30
)

chunks = splitter.split_documents(documents)
print("Total Chunks: ", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"CHUNK {i+1}")
    print(chunk.page_content)
    print("\nMetadat: ", chunk.metadata)
    print("\n")