from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


def embedding_store(vector_store, file_path, chunk_size=100, chunk_overlap=10, separator="\n"):
    loader = TextLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
    chunks = splitter.split_documents(documents)
    vector_store.add_documents(chunks)
