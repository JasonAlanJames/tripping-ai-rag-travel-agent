from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_vector_store():
    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    return vectorstore