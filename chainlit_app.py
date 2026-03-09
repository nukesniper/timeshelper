import os
import chainlit as cl
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
from PIL import Image
from io import BytesIO
from consts import INDEX_NAME

# ---------- SECRETS HELPER ----------
def get_secret(section: str, key: str, default=None):
    """Return secret from chainlit.toml only. Never check environment variables."""
    try:
        import toml
        config_path = os.path.join(os.path.dirname(__file__), ".chainlit", "chainlit.toml")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = toml.load(f)

            if not section:
                if key in config:
                    return str(config[key])
            else:
                if section in config and key in config[section]:
                    return str(config[section][key])

        return default if default is not None else ""
    except Exception as e:
        print(f"Debug: Error loading chainlit.toml: {str(e)}")
        return default if default is not None else ""


# ---------- OPENAI ----------
OPENAI_API_KEY = get_secret("openai", "OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in chainlit.toml")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

OPENAI_PROJECT = get_secret("openai", "OPENAI_PROJECT")
if OPENAI_PROJECT:
    os.environ["OPENAI_PROJECT"] = OPENAI_PROJECT

GLOBAL_OPENAI_KEY = OPENAI_API_KEY


# ---------- PINECONE ----------
PINECONE_API_KEY = get_secret("pinecone", "api_key") or ""
PINECONE_ENV_OR_REGION = (
    get_secret("pinecone", "environment")
    or get_secret("pinecone", "region")
    or ""
)

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

if PINECONE_ENV_OR_REGION:
    os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENV_OR_REGION
    os.environ["PINECONE_REGION"] = PINECONE_ENV_OR_REGION

print(f"Debug: Pinecone API key loaded: {bool(PINECONE_API_KEY)}")
print(f"Debug: Pinecone environment loaded: {bool(PINECONE_ENV_OR_REGION)}")

# Import only AFTER secrets are loaded and exported to os.environ
from backend.core import run_llm

# ========== END OF SETUP ==========

def extract_source_titles(source_paths_tuple: tuple) -> dict[str, str]:
    """
    Use LLM to extract human-readable titles from source file paths.
    Returns mapping of path -> title.
    """
    if not source_paths_tuple:
        return {}
    
    from langchain_openai import ChatOpenAI
    
    api_key = GLOBAL_OPENAI_KEY
    if not api_key:
        print(f"Debug: OPENAI_API_KEY missing in global context")
        return {}
    
    try:
        chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=api_key)
        
        paths_list = sorted(list(source_paths_tuple))
        paths_str = "\n".join(f"- {p}" for p in paths_list)
        
        prompt = f"""Given these source file paths, extract or infer a clean, concise title for each one.
The title should be human-readable and describe what the document is about based on its name and directory.
Keep titles short (2-8 words max).

File paths:
{paths_str}

Respond in this exact format for each path (one per line, matching the order above):
<title>

Example for path "Chapter_5_Thermodynamics.pdf":
Chapter 5: Thermodynamics

Example for path "intro_to_python.md":
Introduction to Python

Now respond with only the titles, one per line, in the same order as the paths above:"""
        
        response = chat.invoke(prompt)
        content = response.content.strip()
        
        titles_list = [line.strip() for line in content.split('\n') if line.strip()]
        
        titles_map = {}
        for i, path in enumerate(paths_list):
            if i < len(titles_list):
                titles_map[path] = titles_list[i]
            else:
                titles_map[path] = os.path.basename(path)
        
        return titles_map
    except Exception as e:
        cl.warning(f"Could not extract titles from sources: {e}")
        return {}

def create_sources_string(source_urls: set[str] | None, titles_map: dict[str, str] | None = None) -> str:
    if not source_urls:
        return ""
    
    if titles_map is None:
        titles_map = {}
    
    lines = []
    for i, src in enumerate(sorted(source_urls), start=1):
        title = titles_map.get(src, os.path.basename(src) or src)
        lines.append(f"{i}. {title}")
    
    return "sources:\n" + "\n".join(lines) + "\n"

def get_profile_picture(email):
    gravatar_url = f"https://www.gravatar.com/avatar/{hash(email)}?d=identicon&s=200"
    response = requests.get(gravatar_url)
    img = Image.open(BytesIO(response.content))
    return img

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat with user profile and session state."""
    user_name = "Lucas Fernandez de Losada"
    user_email = "lucasfer@mit.edu"
    
    profile_pic = Image.open("Profile_Pic_MIT.jpg")
    cl.user_session.set("profile_pic", profile_pic)
    cl.user_session.set("user_name", user_name)
    cl.user_session.set("user_email", user_email)
    
    cl.user_session.set("chat_answers_history", [])
    cl.user_session.set("user_prompt_history", [])
    cl.user_session.set("chat_history", [])
    
    await cl.Message(
        content="Welcome to Synes Nuclear Graph RAG! How can I assist you today?",
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages and generate responses."""
    prompt = message.content
    
    if not prompt:
        return
    
    chat_history = cl.user_session.get("chat_history", [])
    
    # Generate response
    # Ensure OPENAI_API_KEY is set before calling run_llm
    # Use the pre-loaded key
    if not GLOBAL_OPENAI_KEY:
        await cl.Message(
            content="Error: OPENAI_API_KEY validation failed at startup"
        ).send()
        return
    
    generated_response = run_llm(
    query=prompt,
    chat_history=chat_history,
    openai_api_key=GLOBAL_OPENAI_KEY,
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_environment=PINECONE_ENV_OR_REGION,
    )
        
    # Extract sources
    raw_sources = generated_response.get("sources", [])
    if not raw_sources and "context" in generated_response:
        try:
            raw_sources = [getattr(doc, "metadata", {}).get("source") for doc in generated_response["context"]]
            raw_sources = [s for s in raw_sources if s]
        except Exception:
            raw_sources = []
    sources = set(raw_sources)
    
    # Extract titles
    titles_map = extract_source_titles(tuple(sorted(sources))) if sources else {}
    
    formatted_response = f"{generated_response.get('answer', '')} \n\n{create_sources_string(sources, titles_map)}"
    
    # Update session state
    cl.user_session.get("user_prompt_history").append(prompt)
    cl.user_session.get("chat_answers_history").append(formatted_response)
    cl.user_session.get("chat_history").append(("human", prompt))
    cl.user_session.get("chat_history").append(("ai", generated_response["answer"]))
    
    # Send response
    await cl.Message(
        content=formatted_response,
    ).send()