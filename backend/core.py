#from dotenv import load_dotenv

#load_dotenv()
from __future__ import annotations

from typing import Any, Dict, List, Optional
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME


class ConfigError(RuntimeError):
    pass


def _require_env(name: str) -> str:
    """Fetch required env var or raise a clear error."""
    v = os.getenv(name)
    if not v:
        raise ConfigError(
            f"Missing required environment variable: {name}. "
            f"Set it in Streamlit Secrets and export it to env (e.g. os.environ['{name}'] = '...')."
        )
    return v


def run_llm(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    *,
    openai_api_key: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    pinecone_environment: Optional[str] = None,
    index_name: str = INDEX_NAME,
    embedding_model: str = "text-embedding-3-small",
    chat_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the RAG chain. Keys can be passed explicitly or read from env vars:
    - OPENAI_API_KEY
    - PINECONE_API_KEY, PINECONE_ENVIRONMENT
    """

# ---- Keys / env setup ----
api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ConfigError("OPENAI_API_KEY is not set.")

if pinecone_api_key:
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
pc_key = os.getenv("PINECONE_API_KEY")
if not pc_key:
    raise ConfigError("PINECONE_API_KEY is not set. Add it under [pinecone].api_key in Secrets.")

# Accept either ENVIRONMENT or REGION; normalize to ENVIRONMENT
pc_env = pinecone_environment or os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_REGION")
if not pc_env:
    raise ConfigError(
        "PINECONE_ENVIRONMENT (or PINECONE_REGION) is not set. "
        "In Secrets, set [pinecone].environment = \"us-east-1-aws\" (or your region)."
    )
os.environ["PINECONE_ENVIRONMENT"] = pc_env  # ensure downstream libs see it


    # If the app relies on env, ensure they exist (will raise with a clear message)
    _ = os.getenv("PINECONE_API_KEY") or _require_env("PINECONE_API_KEY")
    _ = os.getenv("PINECONE_ENVIRONMENT") or _require_env("PINECONE_ENVIRONMENT")

    # ---- Models ----
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
    chat = ChatOpenAI(model=chat_model, temperature=temperature, api_key=api_key)

    # ---- Vector store / retriever ----
    if namespace:
        docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
    else:
        docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # ---- Prompts & chains ----
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=docsearch.as_retriever(),
        prompt=rephrase_prompt,
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain,
    )

    # ---- Invoke ----
    payload = {
        "input": query,
        "chat_history": chat_history or [],
    }
    result = qa.invoke(input=payload)
    return result


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm2(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(model_name="gpt-4o", verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    rag_chain = (
        {
            "context": docsearch.as_retriever() | format_docs,
            "input": RunnablePassthrough(),
        }
        | retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
    )

    retrieve_docs_chain = (lambda x: x["input"]) | docsearch.as_retriever()

    chain = RunnablePassthrough.assign(context=retrieve_docs_chain).assign(
        answer=rag_chain
    )

    result = chain.invoke({"input": query, "chat_history": chat_history})
    return result