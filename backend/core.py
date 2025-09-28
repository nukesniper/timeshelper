# backend/core.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

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


def _openai_sanity_check(api_key: str) -> None:
    """Fail fast with a readable error if key/project/org is wrong."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key.strip(),
        organization=(os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION") or "").strip() or None,
        project=(os.getenv("OPENAI_PROJECT") or "").strip() or None,
    )
    try:
        _ = client.models.list()  # verify credentials + headers
    except Exception as e:
        raise ConfigError(
            "OpenAI authentication failed. Double-check that:\n"
            "• OPENAI_API_KEY starts with sk-proj- and belongs to THIS project\n"
            "• OPENAI_PROJECT is exactly the 'proj_…' of the same project\n"
            "• (Optional) OPENAI_ORG_ID matches your account’s org, or leave it unset\n"
            "• No conflicting OPENAI_BASE_URL/OPENAI_API_BASE env vars are set\n"
            f"Raw error type: {type(e).__name__}. Check Streamlit Cloud logs for details."
        ) from e


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
    api_key = (openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
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

    # ---- Fast auth sanity (will raise ConfigError with guidance) ----
    _openai_sanity_check(api_key)

    # ---- Models ----
    # Read optional Organization / Project for project-scoped keys (sk-proj-*)
    org_id = (os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION") or "").strip() or None
    project = (os.getenv("OPENAI_PROJECT") or "").strip() or None

    # Basic sanity on the key format
    if not api_key.startswith("sk-") or len(api_key) < 20:
        raise ConfigError(
            "OPENAI_API_KEY looks invalid. Make sure it starts with 'sk-' "
            "and is copied exactly from the OpenAI dashboard."
        )

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

    # ---- Vector store / retriever ----
    if namespace:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
    else:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ---- Prompts & chains ----
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    history_aware = create_history_aware_retriever(chat, retriever, rephrase_prompt)
    stuff_chain = create_stuff_documents_chain(chat, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware, stuff_chain)

    # ---- Invoke ----
    result = rag_chain.invoke({"input": query, "chat_history": chat_history or []})

    # LangChain may return "answer" or "output_text" depending on versions
    answer = result.get("answer") or result.get("output_text") or ""
    context = result.get("context", []) or []
    sources: List[str] = []
    for d in context:
        try:
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source") or meta.get("file") or meta.get("url")
            if src:
                sources.append(str(src))
        except Exception:
            pass

    return {"answer": answer, "sources": sources, "k": len(context)}
