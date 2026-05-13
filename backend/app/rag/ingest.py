from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.rag.vector_store import get_vector_store

def ingest_documents():
    vectorstore = get_vector_store()

    docs = [
        Document(
            page_content="""
Tokyo Travel Guide:
- Best time to visit: March-May, September-November
- Popular areas: Shibuya, Shinjuku, Asakusa
- Must try food: Sushi, Ramen, Tempura
- Transportation: JR Pass, Suica card
"""
        ),
        Document(
            page_content="""
Japan Travel Tips:
- Respect quiet public spaces
- Carry cash
- Learn basic phrases
- Trains are extremely punctual
"""
        )
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    vectorstore.add_documents(split_docs)
    vectorstore.persist()

    print(f"Ingested {len(split_docs)} chunks")