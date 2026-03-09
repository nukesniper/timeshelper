import os
import streamlit as st
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import requests
import json
import chainlit as cl

# ---------- SECRETS HELPER ----------
def get_secret(section: str, key: str, default=None):
    """Return st.secrets[section][key] if present, else env var `key`, else default."""
    try:
        if section in st.secrets and key in st.secrets[section]:
            v = st.secrets[section][key]
            if v is not None:
                return str(v)
    except Exception:
        pass
    return os.environ.get(key, default)

# ---------- SLACK ----------
def send_slack_report(subject: str, body: str) -> bool:
    slack_token = get_secret("slack", "SLACK_API_TOKEN")       # expects xoxb-...
    slack_channel_id = get_secret("slack", "SLACK_CHANNEL_ID") # e.g. C0123456789

    if not slack_token or not slack_channel_id:
        st.error("Slack token or channel id missing.")
        return False

    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {slack_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "channel": slack_channel_id,
        "text": f"*{subject}*\n{body}",
        # "username": "App Reporter"   # optional display name for webhooks; ignored for bot tokens
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        data = resp.json()
        if not data.get("ok"):
            st.error(f"Slack API error: {data.get('error')}")
            return False
        return True
    except requests.RequestException as e:
        st.error(f"Failed to reach Slack: {e}")
        return False

def report_issue(subject: str, body: str):
    if send_slack_report(subject, body):
        st.success("Report sent to Slack.")
    else:
        st.error("Failed to send report to Slack.")

# --- OpenAI Org/Project (optional but required for sk-proj- keys) ---
OPENAI_PROJECT = (st.secrets.get("OPENAI_PROJECT") or os.getenv("OPENAI_PROJECT") or "").strip()
OPENAI_ORG_ID = (st.secrets.get("OPENAI_ORG_ID") or os.getenv("OPENAI_ORG_ID") or "").strip()

if OPENAI_PROJECT:
    os.environ["OPENAI_PROJECT"] = OPENAI_PROJECT

if OPENAI_ORG_ID:
    os.environ["OPENAI_ORG_ID"] = OPENAI_ORG_ID



# PINECONE (support both flat and [pinecone] section)
pinecone_section = {}
try:
    pinecone_section = dict(st.secrets.get("pinecone", {}))  # may be empty
except Exception:
    pinecone_section = {}

PINECONE_API_KEY = (
    pinecone_section.get("api_key")
    or get_secret("PINECONE_API_KEY", env="PINECONE_API_KEY")
)

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = str(PINECONE_API_KEY)

# Environment (classic) or region (serverless)
PINECONE_ENV_OR_REGION = (
    pinecone_section.get("environment")
    or pinecone_section.get("region")
    or get_secret("PINECONE_ENVIRONMENT", env="PINECONE_ENVIRONMENT")
    or get_secret("PINECONE_REGION", env="PINECONE_REGION")
)

if PINECONE_ENV_OR_REGION:
    # We export to PINECONE_ENVIRONMENT for backward-compat with classic clients.
    os.environ["PINECONE_ENVIRONMENT"] = str(PINECONE_ENV_OR_REGION)

# ========== END OF SETUP ==========

# Keep imports at top-level and unindented; if this fails, show a friendly error.
try:
    from backend.core import run_llm  # noqa: E402
except Exception as e:
    st.error(
        "Failed to import `run_llm` from `backend.core`. "
        "Check that `backend/__init__.py` is valid Python (no stray `[theme]`), "
        "`backend/core.py` exists, and the repo path is correct."
    )
    st.exception(e)
    st.stop()


# Other imports
from PIL import Image
import requests
from io import BytesIO


@st.cache_data
def extract_source_titles(source_paths_tuple: tuple) -> dict[str, str]:
    """
    Use LLM to extract human-readable titles from source file paths.
    Returns mapping of path -> title.
    Cached to avoid repeated LLM calls for the same sources.
    """
    if not source_paths_tuple:
        return {}
    
    from langchain_openai import ChatOpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
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
        
        # Create mapping ensuring we have enough titles
        titles_map = {}
        for i, path in enumerate(paths_list):
            if i < len(titles_list):
                titles_map[path] = titles_list[i]
            else:
                # Fallback: extract filename from path
                titles_map[path] = os.path.basename(path)
        
        return titles_map
    except Exception as e:
        st.warning(f"Could not extract titles from sources: {e}")
        return {}


def create_sources_string(source_urls: set[str] | None, titles_map: dict[str, str] | None = None) -> str:
    if not source_urls:
        return ""
    
    if titles_map is None:
        titles_map = {}
    
    lines = []
    for i, src in enumerate(sorted(source_urls), start=1):
        # Use LLM-generated title if available, otherwise use filename
        title = titles_map.get(src, os.path.basename(src) or src)
        lines.append(f"{i}. {title}")
    
    return "sources:\n" + "\n".join(lines) + "\n"



# Add this function to get a profile picture
def get_profile_picture(email):
    # This uses Gravatar to get a profile picture based on email
    # You can replace this with a different service or use a default image
    gravatar_url = f"https://www.gravatar.com/avatar/{hash(email)}?d=identicon&s=200"
    response = requests.get(gravatar_url)
    img = Image.open(BytesIO(response.content))
    return img


# Custom CSS for blue theme and modern look
st.markdown(
    """
<style>
    .stApp {
        background-color: #1a237e;  /* Dark blue background */
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #283593;  /* Slightly lighter blue for input */
        color: #FFFFFF;
    }
    .stButton > button {
        background-color: #2196F3;  /* Material blue for buttons */
        color: #FFFFFF;
    }
    .stSidebar {
        background-color: #0d47a1;  /* Darker blue for sidebar */
    }
    .stMessage {
        background-color: #1e88e5;  /* Light blue for messages */
    }
</style>
""",
    unsafe_allow_html=True,
)

# Set page config at the very beginning


# Sidebar user information
with st.sidebar:
    st.title("User Profile")

    # You can replace these with actual user data
    user_name = "Lucas Fernandez de Losada"
    user_email = "lucasfer@mit.edu"

    profile_pic = Image.open("Profile_Pic_MIT.jpg")
    st.image(profile_pic, width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")

st.header("📚 Synes Nuclear Graph Rag")

# Initialize session state
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# Create two columns for a more modern layout
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_input("Prompt", placeholder="Enter your message here...")

with col2:
    if st.button("Submit", key="submit"):
        prompt = prompt or "Hello"  # Default message if input is empty

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        # Prefer `sources` returned by run_llm; fall back to any `context` documents
        raw_sources = generated_response.get("sources") or []
        if not raw_sources and "context" in generated_response:
            try:
                raw_sources = [getattr(doc, "metadata", {}).get("source") for doc in generated_response["context"]]
                raw_sources = [s for s in raw_sources if s]
            except Exception:
                raw_sources = []
        sources = set(raw_sources)

        # Extract LLM-generated titles from source paths
        titles_map = extract_source_titles(tuple(sorted(sources))) if sources else {}
        
        formatted_response = f"{generated_response.get('answer', '')} \n\n{create_sources_string(sources, titles_map)}"

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))

# Display chat history
if st.session_state["chat_answers_history"]:
    # Reverse the order of messages by using reversed() on the zipped lists
    for generated_response, user_query in zip(
        reversed(st.session_state["chat_answers_history"]),
        reversed(st.session_state["user_prompt_history"]),
    ):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        
        # Display the user message with the role 'user'
        st.chat_message("user").markdown(user_query)
        
        # Display the bot message with the role 'bot'
        st.chat_message("bot").markdown(generated_response)

# ---------- REPORT FORM ----------
with st.expander("📨 Report an Issue", expanded=False):
    st.write("If something's not working or you want to suggest improvements, let me know.")

    with st.form("issue_form"):
        user_report_email = st.text_input("Your email (optional)")
        issue_message = st.text_area("Describe the issue")
        submit_report = st.form_submit_button("Send Report")

    if submit_report:
        issue = issue_message.strip()
        if not issue:
            st.warning("Please describe the issue before submitting.")
        else:
            subject = "New Issue Report from Streamlit App"
            body = f"From: {user_report_email or 'Not provided'}\n\nIssue:\n{issue}"

            # --- Prefer webhook if present; else use bot token ---
            webhook_url = get_secret("slack", "SLACK_WEBHOOK_URL")
            if webhook_url:
                # Incoming Webhook path (simplest)
                try:
                    r = requests.post(
                        webhook_url,
                        headers={"Content-Type": "application/json"},
                        data=json.dumps({"text": f"*{subject}*\n{body}"})
                    )
                    r.raise_for_status()
                    st.success("✅ Your report has been sent to Slack. Thank you!")
                except requests.RequestException as e:
                    st.error(f"Failed to send Slack webhook: {e}")
            else:
                # Bot token path
                sent = send_slack_report(subject, body)  # uses xoxb token + channel ID
                if sent:
                    st.success("✅ Your report has been sent to Slack. Thank you!")
                else:
                    st.error("Couldn't send the report to Slack. Please try again later.")


# Add a footer
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit. Courtesy of Eden Marco")
