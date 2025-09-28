#from dotenv import load_dotenv

#load_dotenv()
# backend/core.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

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
    """Config/Env problems raised with a clear message."""
    pass


def _require_env(name: str) -> str:
    """Fetch required env var or raise a clear error."""
    v = os.getenv(name)
    if not v:
        raise ConfigError(
            f"Missing required environment variable: {name}. "
            f"Set it in Streamlit Secrets and export it to env "
            f"(e.g., os.environ['{name}'] = '...')."
        )
    return v


def run_llm(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    *,
    openai_api_key: Optional[str] = None,
    pinecone_api_key: Optional[str] = None,
    pinecone_environment: Optional[str] = None,  # for classic pods; serverless uses region
    index_name: str = INDEX_NAME,
    embedding_model: str = "text-embedding-3-small",
    chat_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the RAG chain. Keys can be passed explicitly or read from env vars:
      - OPENAI_API_KEY
      - PINECONE_API_KEY, PINECONE_ENVIRONMENT (or PINECONE_REGION)
    Returns:
      dict(answer=str, sources=list[str], k=int)
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

    # Accept either ENVIRONMENT or REGION; normalize to ENVIRONMENT for libs that still expect it
    pc_env = pinecone_environment or os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_REGION")
    if not pc_env:
        raise ConfigError(
            "PINECONE_ENVIRONMENT (or PINECONE_REGION) is not set. "
            'In Secrets, set [pinecone].environment = "gcp-starter" (classic) '
            'or a serverless region like "us-east-1-aws".'
        )
    os.environ["PINECONE_ENVIRONMENT"] = pc_env  # ensure downstream libs see it

        # ---- Models ----
    # Read optional Organization / Project for project-scoped keys (sk-proj-*)
    org_id = (os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION") or "").strip() or None
    project = (os.getenv("OPENAI_PROJECT") or "").strip() or None

    # Basic sanity on the key to catch empty/whitespace values early
    api_key = api_key.strip()
    if not api_key.startswith("sk-") or len(api_key) < 20:
        raise ConfigError(
            "OPENAI_API_KEY looks invalid. Make sure it starts with 'sk-' "
            "and is copied exactly from the OpenAI dashboard."
        )

    # Build clients, forwarding optional org/project
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        organization=org_id,
        project=project,
    )
    chat = ChatOpenAI(
        model=chat_model,
        temperature=temperature,
        api_key=api_key,
        organization=org_id,
        project=project,
    )

    # ---- Prompts & chains ----
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    history_aware = create_history_aware_retriever(chat, retriever, rephrase_prompt)
    stuff_chain = create_stuff_documents_chain(chat, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware, stuff_chain)

    # ---- Invoke ----
    result = rag_chain.invoke(
        {"input": query, "chat_history": chat_history or []}
    )

    # LangChain may return "answer" or "output_text" depending on versions
    answer = result.get("answer") or result.get("output_text") or ""
    context = result.get("context", []) or []
    sources = []
    for d in context:
        try:
            # try common metadata keys
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source") or meta.get("file") or meta.get("url")
            if src:
                sources.append(str(src))
        except Exception:
            pass

    return {"answer": answer, "sources": sources, "k": len(context)}

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