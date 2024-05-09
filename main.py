from langchain.retrievers import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres import PGVector
from fastapi import FastAPI
from pydantic import BaseModel
from Operation.embedding import embedding_store
from Operation.chat import Chat
from langchain_community.chat_message_histories import PostgresChatMessageHistory

embeddings = OllamaEmbeddings(model="llama3:latest")
collection_name = "docs"
connection = "postgresql+psycopg://admin:123@localhost:5432/embeddings"  # Uses psycopg3!

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True
)

retriever = vectorstore.as_retriever(k=10)

# template = """You are an assistant for question-answering tasks.
# Use the following pieces of retrieved context to answer the question.
# If you don't know the answer, just say that you don't know.
# Use three sentences maximum and keep the answer concise.
# Question: {question}
# Context: {context}
# Answer:
# """
#

llm = ChatOllama(model="llama3:latest", temperature=0.2)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm
)

history_retriever = lambda session_id: PostgresChatMessageHistory(
    session_id=session_id,
    connection_string="postgresql://admin:123@localhost:5432/chat_history",
)

app = FastAPI()

baseURL = "/api/v1"


class Ask(BaseModel):
    input: str
@app.post(f"{baseURL}/ask")
def rag_query(body: Ask):
    chat = Chat(llm, retriever_from_llm, history_retriever)
    data = chat.run(
        input={"input": body.input},
        config={"configurable": {"session_id": "abc123"}}
    )
    print(data)
    for chat in data["chat_history"]:
        print(chat)
    return {"data": data}


@app.post(f"{baseURL}/embedding")
def embedding_api():
    embedding_store(vectorstore, "./files/劳动法.txt")
    return {"data": "success"}


class History(BaseModel):
    session_id: str

@app.post(f"{baseURL}/history")
def queryChatHistoryList(body: History):
    print(history_retriever(body.session_id).messages)
    return {"data": history_retriever(body.session_id).messages}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8080)
