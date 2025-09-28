import os
import streamlit as st

# ========== PAGE CONFIG (first Streamlit call) ==========
st.set_page_config(page_title="Your App Title", page_icon="🧊", layout="wide")

# ========== SECRETS HELPER ==========
def get_secret(name: str, env: str | None = None, default=None):
    """Return a secret from st.secrets[name] (flat), else os.environ[env], else default."""
    try:
        if name in st.secrets:
            v = st.secrets[name]
            if v is not None:
                return str(v)
    except Exception:
        pass
    if env and env in os.environ:
        return os.environ[env]
    return default

# ========== LOAD KEYS & EXPORT TO ENV ==========
# OPENAI
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", env="OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    st.sidebar.success("Secrets loaded. Found: OPENAI_API_KEY")
else:
    st.sidebar.error("OPENAI_API_KEY not found in Secrets or environment.")

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
    st.sidebar.success("Secrets loaded. Found: PINECONE_API_KEY")
else:
    st.sidebar.error("PINECONE_API_KEY missing (either flat or [pinecone].api_key).")

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
    st.sidebar.success(f'PINECONE_ENVIRONMENT set: {PINECONE_ENV_OR_REGION}')
else:
    st.sidebar.warning("No Pinecone environment/region provided (ok for some serverless setups).")

# Optional sanity
st.sidebar.caption("🔐 Pinecone sanity")
st.sidebar.write({
    "PINECONE_API_KEY": bool(os.getenv("PINECONE_API_KEY")),
    "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT"),
})

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


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


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
    user_email = "lucas.fernandez-de-losada@psi.ch"

    profile_pic = Image.open("Profile_Pic_MIT.jpg")
    st.image(profile_pic, width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")

st.header("Lucas PhD Documentation Helper")

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

        sources = set(doc.metadata["source"] for doc in generated_response["context"])
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )

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
        message(user_query, is_user=True, key=f"user_{timestamp}_{hash(user_query)}")
        message(generated_response, key=f"bot_{timestamp}_{hash(generated_response)}")

# Add a footer
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")
