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
    import re

    key = (api_key or "").strip()
    org = (os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION") or "").strip() or None
    proj = (os.getenv("OPENAI_PROJECT") or "").strip() or None
    base = (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "").strip() or None

    # Quick validations
    if key.startswith("sk-proj-") and not (proj and proj.startswith("proj_")):
        raise ConfigError(
            "Using a project-scoped key (sk-proj-*) but OPENAI_PROJECT is missing or invalid.\n"
            "Set OPENAI_PROJECT to your exact ID like 'proj_xxxxx' from the SAME project that issued the key."
        )
    if base:
        raise ConfigError(
            f"Unexpected OPENAI_BASE_URL/OPENAI_API_BASE is set ({base}). "
            "Unset it unless you intentionally use Azure/proxy."
        )

    def _mask(s: str | None, keep: int = 6) -> str:
        if not s: return "(unset)"
        return ("*" * max(len(s) - keep, 0)) + s[-keep:]

    # Try client WITHOUT org first (project-scoped keys normally don't need org)
    def try_client(_org: str | None):
        return OpenAI(api_key=key, organization=_org, project=proj)

    last_err = None
    for _org_try in (None, org):
        try:
            client = try_client(_org_try)
            client.models.list()  # cheap auth-required call
            return  # success!
        except Exception as e:
            last_err = e

    # If we got here, both attempts failed
    hint = (
        "OpenAI authentication failed.\n"
        "Checklist:\n"
        "  • OPENAI_API_KEY must be valid. For project keys it starts with 'sk-proj-'.\n"
        "  • OPENAI_PROJECT must be the exact 'proj_…' from the SAME project as the key.\n"
        "  • Remove OPENAI_ORG_ID unless you are sure it matches the project’s org.\n"
        "  • Ensure OPENAI_BASE_URL / OPENAI_API_BASE is NOT set (unless using Azure/proxy).\n"
        f"Used values:\n"
        f"  • API key:   {_mask(key)}\n"
        f"  • Project:   {proj or '(unset)'}\n"
        f"  • Org ID:    {org or '(unset)'}\n"
        f"  • Base URL:  {base or '(default)'}\n"
        f"Raw error type: {type(last_err).__name__}. Check Cloud logs for the server message."
    )
    raise ConfigError(hint) from last_err


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
    project = (os.getenv("OPENAI_PROJECT") or "").strip() or None
    org_id = (os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION") or "").strip() or None

    api_key = api_key.strip()
    if not api_key.startswith("sk-") or len(api_key) < 20:
        raise ConfigError("OPENAI_API_KEY looks invalid. Copy it exactly from the dashboard.")

    # Do the sanity check BEFORE constructing LangChain wrappers
    _openai_sanity_check(api_key)

    # Build clients; only pass organization if you really set it
    emb_kwargs = {"model": embedding_model, "api_key": api_key, "project": project}
    chat_kwargs = {"model": chat_model, "temperature": temperature, "api_key": api_key, "project": project}
    if org_id:
        emb_kwargs["organization"] = org_id
        chat_kwargs["organization"] = org_id

    embeddings = OpenAIEmbeddings(**emb_kwargs)
    chat = ChatOpenAI(**chat_kwargs)

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
