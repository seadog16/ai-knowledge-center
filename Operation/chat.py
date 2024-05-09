from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


class Chat:
    def __init__(self, llm, retriever, history_retriever):
        self.history_retriever = history_retriever
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
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
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run(self, **kw):
        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.history_retriever,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        return conversational_rag_chain.invoke(**kw)
