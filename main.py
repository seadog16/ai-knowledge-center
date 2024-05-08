import json

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_postgres import PGVector
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from fastapi import FastAPI
from pydantic import BaseModel

# embeddings = OllamaEmbeddings(model="qwen:14b", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="llama3:latest")
collection_name = "docs"
connection = "postgresql+psycopg://admin:123@localhost:5432/embeddings"  # Uses psycopg3!

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True
)


# loader = TextLoader("./files/劳动法.txt")
# documents = loader.load()
# splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10, separator="\n")
# chunks = splitter.split_documents(documents)
# print("===拆字完毕===")
# print(len(chunks))
# for c in chunks:
#     print(c)
# vectorstore.add_documents(chunks)
# print("===灌库完毕===")
#
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
# template = """
# 你是一个智能助手，请总结知识库的内容来回答问题，请列举知识库中的数据详细回答。
# 当所有知识库内容都与问题无关时，你的回答必须包括“知识库中未找到您要的答案！”这句话。
# 回答需要考虑聊天历史。
#
# 用户问题: {question}
# 知识库的内容: {context}
# 用中文回答:
# """
# prompt = ChatPromptTemplate.from_template(template)

# llm = ChatOllama(model="qwen:14b", base_url="http://localhost:11434", temperature=0)
llm = ChatOllama(model="llama3:latest", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

contextualize_q_system_prompt ="""Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.\
    Answer in Chinese."""

# contextualize_q_system_prompt = """给定一个聊天记录和最新的用户问题\
#     哪一个可能在聊天历史中引用上下文，形成一个独立的问题\
#     不用聊天记录也能看懂。不要回答这个问题，\
#     如果需要，只需重新制定它，否则就原样返回。\
#     用中文回答"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """你是一个智能助手，请总结知识库的内容来回答问题，请列举知识库中的数据详细回答。
当所有知识库内容都与问题无关时，你的回答必须包括“知识库中未找到您要的答案！”这句话。
回答需要考虑聊天历史。用中文回答

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
chat_history =[]
app = FastAPI()

class Body(BaseModel):
    input: str
@app.post("/api/v1/ask")
def rag_query(body: Body):
    data = rag_chain.invoke({"input": body.input, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=body.input), data["answer"]])
    for document in data["context"]:
        print(document)
        print()
    return {"data": data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8080)