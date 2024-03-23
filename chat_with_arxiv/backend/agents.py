import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


def response_from_retrieval_chain(model, prompt, retriever, chat_history):
    template = ChatPromptTemplate.from_template(
        """You are an expert researcher and personal assist.
Your task is to answer the questions about the arxiv paper with these instructions.
Your taks is to provide clear, concise, and nice responses, along with explanations and annotations.
You must generate answers based on the given <context>.

<context>
{context}
</context>

Question: {input}
"""
    )
    llm = ChatOllama(base_url=os.environ.get("OLLAMA_API_URL"), model=model)
    doc_chain = create_stuff_documents_chain(llm, template)

    history = []
    for _log in chat_history:
        if _log["role"] == "user":
            msg = HumanMessage(content=_log["content"])
        else:
            msg = AIMessage(content=_log["content"])
        history.append(msg)

    _input = {"input": prompt, "chat_history": history}
    retriever_chain = create_retrieval_chain(retriever, doc_chain)

    for chunk in retriever_chain.stream(_input):
        if "answer" in chunk:
            yield chunk["answer"]


def summarizer(model, paper_metadata: dict[str, None]):
    template = ChatPromptTemplate.from_template(
        """You are an expert researcher and personal assist.
Your task is to answer the questions about the arxiv paper with these instructions.
Your taks is to provide clear, concise, and nice responses, along with explanations and annotations.
But First, You must summarize the given abstract of the paper in <context> using 3-5 sentences.
Below is the items you need to include in your summary.

<items which need to include in your summary>
1. The purpose of the research.
2. The contributions of the research.
3. What are the main findings and conclusions from the paper.
</items which need to include in your summary>

<context>
{context}
</context>

Summary:
"""
    )
    abstract = paper_metadata["Summary"]
    llm = ChatOllama(base_url=os.environ.get("OLLAMA_API_URL"), model=model)
    for chunk in llm.stream(template.format(context=abstract)):
        yield chunk.content
