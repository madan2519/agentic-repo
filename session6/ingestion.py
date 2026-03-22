from dotenv import load_dotenv
import os

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def main():
    print("Ingesting...")
    loader = TextLoader("wikicontent.txt", encoding="utf8")
    # print("loader:::", loader)
    document = loader.load()
    # print("document:::", document)
    # print(document)

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # print("text_splitter:::", text_splitter)
    chunks = text_splitter.split_documents(document)
    # print("chunks:::", chunks)

    print("Embedding...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    # print("embeddings:::", embeddings)

    print("PineconeVectorStore...")
    vectorstore = PineconeVectorStore.from_documents(index_name=os.getenv("INDEX_NAME"), embedding=embeddings, documents=chunks)
    # print("vectorstore:::", vectorstore)
    vectorstore.add_documents(chunks)
    print("Done")


if __name__ == "__main__":
    main()