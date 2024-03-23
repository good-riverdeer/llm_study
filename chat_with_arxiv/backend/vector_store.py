import os

from langchain_core.documents import Document
from langchain_community.document_loaders.arxiv import ArxivLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.cached_chroma import CachedChroma


def filter_metadata_types(item) -> bool:
    if isinstance(item, str):
        return True
    if isinstance(item, int):
        return True
    if isinstance(item, float):
        return True
    if isinstance(item, bool):
        return True
    return False


def load_arxiv_doc(query) -> tuple[list[Document], dict[str, str]]:
    loader = ArxivLoader(query=query, load_max_docs=1, load_all_available_meta=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    doc = loader.load()[0]
    splited_content = splitter.split_text(doc.page_content)
    metadata = {k: v for k, v in doc.metadata.items() if filter_metadata_types(v)}
    new_docs = [
        Document(page_content=content, metadata=metadata) for content in splited_content
    ]
    return new_docs, doc.metadata


def store_docs(docs: list[Document], model_name: str, collection_name: str):
    embeddings = OllamaEmbeddings(
        base_url=os.environ["OLLAMA_API_URL"],
        model=model_name,
    )
    db = CachedChroma.from_documents_with_cache(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./data",
    )
    return db
