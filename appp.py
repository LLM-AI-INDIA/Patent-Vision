import os, json, base64, re
import streamlit as st

SECRET_KEYS = ("gcp_service_account","GCP_SERVICE_ACCOUNT","gcp_sa","gcp_service_account_b64")
sa_json = None

def clean_control_chars(s: str) -> str:
    # Remove ASCII control chars except newline (10) and tab (9) and carriage return (13 -> normalized)
    # Normalize CRLF to LF
    s = s.replace('\r\n','\n').replace('\r','\n')
    # Remove other control chars
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)

# 1) If secret stored as proper JSON object (Streamlit may load as dict), convert to string
if "gcp_service_account" in st.secrets:
    val = st.secrets["gcp_service_account"]
    if isinstance(val, dict):
        sa_json = json.dumps(val, ensure_ascii=False)
    else:
        sa_json = str(val)

# 2) Base64 fallback
if (not sa_json) and ("gcp_service_account_b64" in st.secrets):
    try:
        decoded = base64.b64decode(st.secrets["gcp_service_account_b64"]).decode("utf-8")
        sa_json = decoded
    except Exception as ex:
        st.error("Failed to decode base64 GCP secret: " + str(ex))

# 3) generic loop for other names (if user used different key)
if not sa_json:
    for k in SECRET_KEYS:
        if k in st.secrets:
            v = st.secrets[k]
            sa_json = json.dumps(v) if isinstance(v, dict) else str(v)
            break

# 4) sanitize + validate JSON before writing
if sa_json:
    sa_json = clean_control_chars(sa_json)
    # If the JSON was double-encoded (a JSON string containing JSON), try to unwrap:
    try:
        parsed = json.loads(sa_json)
        # if parsed is a string (i.e., double-encoded), try to decode again
        if isinstance(parsed, str):
            parsed2 = json.loads(parsed)
            sa_json = json.dumps(parsed2, ensure_ascii=False)
        else:
            sa_json = json.dumps(parsed, ensure_ascii=False)
    except Exception as ex:
        st.error("GCP secret appears invalid JSON after cleaning: " + str(ex))
        st.stop()

    # write to /tmp and set env var
    cred_path = "/tmp/gcp_service_account.json"
    try:
        with open(cred_path, "w", encoding="utf-8") as fh:
            fh.write(sa_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
        #st.info("GCP credentials loaded from Streamlit secrets.")
    except Exception as e:
        st.error("Failed to write service account JSON to /tmp: " + str(e))
else:
    st.warning("No GCP service account found in Streamlit secrets. Add 'gcp_service_account' or 'gcp_service_account_b64'.")


# appp_chat_with_dropdowns.py
# Patent RAG — Chat (dropdown controls)
# Run: streamlit run appp_chat_with_dropdowns.py

import streamlit as st
from google.cloud import bigquery
import pandas as pd
import datetime
import html as html_lib
import json
import re
import traceback
import os
import base64

# ---------- CONFIG ----------
PROJECT = "genai-poc-424806"  # <<--- change this if needed
DATASET = "patent_demo"
EMB_MODEL = f"`{PROJECT}.{DATASET}.embedding_model`"
LLM_MODEL = f"`{PROJECT}.{DATASET}.gemini_text`"
EMB_TABLE = f"`{PROJECT}.{DATASET}.patent_embeddings`"
DEFAULT_LOCATION = "US"
RUN_BQ_TEST = True

st.set_page_config(page_title="Patent RAG — Chat", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
/* ---------- Sidebar sizing and color ---------- */
section[data-testid="stSidebar"], div[data-testid="stSidebar"] {
  background-color: #318CE7 !important;
  padding: 0rem 0.2rem !important;
  color: #ffffff !important;
  width: 335px !important;         /* <-- increased */
  min-width: 335px !important;     /* <-- increased */
  box-sizing: border-box !important;
}

/* Header */
header[data-testid="stHeader"] { 
  visibility: visible !important;
  display: flex !important;
  height: 60px !important;   /* enough room for buttons */
  padding: 0 1rem !important;
  z-index: 9999 !important;  /* keep it above other elements */
  }

/* Centered inline logo id */
#custom-centered-logo {
  display:block !important;
  margin-left:auto !important;
  margin-right:auto !important;
  width: 260px !important;
  max-width: 60% !important;
  position: relative !important;
  top: -50px !important;
  margin-top: 0 !important;   /* kill extra margin */
  z-index: 9999 !important;
}

/* Reduce default top padding/margin of the whole app */
div[data-testid="stAppViewContainer"] {
  padding-top: 0 !important;     /* remove top padding */
  margin-top: 0 !important;  /* pull everything up */
}

/* Also adjust the main block container */
div[data-testid="stAppViewBlockContainer"],
div.block-container {
  margin-top: -60px !important;   /* adjust as needed */
  padding-top: 2 !important;
}



/* Centered heading */
.centered-h2 { text-align:center; margin-top: -20px; font-size: 28px !important; color: #222222 !important; }

/* Main Send button green (applies to st.form submit) */
.stButton > button {
  background-color: #32CD32 !important;
  color: #ffffff !important;
  border: none !important;
  padding: 10px 14px !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}

/* Sidebar buttons (white) */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] button {
  background-color: #ffffff !important;
  color: #222222 !important;
  border: 1px solid rgba(0,0,0,0.12) !important;
  padding: 6px 10px !important;
  border-radius: 6px !important;
  box-shadow: none !important;
  font-weight: 700 !important;
}

/* Make ONLY the Clear chat form submit green (targets the exact input value) */
section[data-testid="stSidebar"] input[type="submit"][value="Clear chat"] {
  background-color: #32CD32 !important;
  color: #ffffff !important;
  border: none !important;
  padding: 8px 12px !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
  box-shadow: none !important;
}

/* Chat bubble styles */
.user-bubble {
  background: #e6e6e6 !important;   /* user = light grey */
  color: #222222 !important;
  padding: 12px !important;
  border-radius: 12px !important;
  white-space: pre-wrap !important;
}
.assistant-bubble {
  background: #ffffff !important;   /* assistant = white */
  color: #222222 !important;
  padding: 12px !important;
  border-radius: 12px !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  white-space: pre-wrap !important;
}

/* Make selectboxes / inputs visually white and tidy in sidebar */
section[data-testid="stSidebar"] select, section[data-testid="stSidebar"] .stSelectbox, section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea {
  background: #ffffff !important;
  color: #222222 !important;
  border-radius: 8px !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
}

/* Make BigQuery test appear full-width and not wrap */
section[data-testid="stSidebar"] .stButton > button.full-width {
  width:100% !important;
  text-align:left !important;
  padding-left:12px !important;
  white-space: normal !important;
}

/* remove odd extra margins around column buttons */
section[data-testid="stSidebar"] .stButton { margin:0 !important; padding:0 !important; }

/* ensure inputs/textarea are white */
input[type="number"],
textarea, input[type="text"], select {
  background: #ffffff !important;
  color: #222222 !important;
  border-radius: 6px !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
}

/* hide footer */
footer { visibility: hidden !important; height: 0px !important; }

@media (max-width: 1200px) {
  section[data-testid="stSidebar"] { width: 300px !important; min-width:300px !important; }
  #custom-centered-logo { top: -30px !important; width: 240px !important; max-width:55% !important; margin-bottom: 0 !important; }
}

div[aria-label="clear_chat_button"] > button {
  background-color: #32CD32 !important;
  color: #ffffff !important;
  border: none !important;
  padding: 8px 12px !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
  box-shadow: none !important;
}

div[data-testid="stAppViewContainer"] img#custom-centered-logo {
  margin-bottom: -20px !important;   /* pull heading closer to logo */
}

div[data-testid="stAppViewContainer"] h2.centered-h2 {
  margin-top: -20px !important;      /* reduce gap above heading */
}

/* Reduce space below the heading */
.centered-h2 {
  margin-bottom: 4px !important;   /* tighter gap under "Patent Vision" */
}

/* Reduce space above the description text */
div[data-testid="stAppViewContainer"] p {
  margin-top: 0 !important;        /* remove extra top space */
}

section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    display: block !important;
    color: #888 !important;   /* make arrow grey */
}

/* Collapse button — completely transparent */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] * {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    outline: none !important;
}

/* On hover/focus/active — still transparent */
[data-testid="stSidebarCollapseButton"]:hover,
[data-testid="stSidebarCollapseButton"]:focus,
[data-testid="stSidebarCollapseButton"]:active,
[data-testid="stSidebarCollapseButton"] *:hover,
[data-testid="stSidebarCollapseButton"] *:focus,
[data-testid="stSidebarCollapseButton"] *:active {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    outline: none !important;
}

/* Force the collapse button itself transparent */
button[data-testid="stSidebarCollapseButton"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    outline: none !important;
}

/* Kill hover, focus, active states */
button[data-testid="stSidebarCollapseButton"]:hover,
button[data-testid="stSidebarCollapseButton"]:focus,
button[data-testid="stSidebarCollapseButton"]:active {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    outline: none !important;
}

/* Optional: recolor the arrow icon (white to match sidebar) */
button[data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
    color: #ffffff !important;
}

/* -----------------------
   Collapse-arrow: absolutely transparent (no white hover box)
   Insert this at the *end* of your CSS block so it overrides everything.
   ----------------------- */

/* target the button, its descendants and common ancestor placements */
[data-testid="stSidebarCollapseButton"],
header [data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] * ,
header [data-testid="stSidebarCollapseButton"] * ,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] * {
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  box-shadow: none !important;
  -webkit-box-shadow: none !important;
  border: none !important;
  outline: none !important;
  filter: none !important;
  backdrop-filter: none !important;
}

/* also clear pseudo-elements that create hover rectangles */
[data-testid="stSidebarCollapseButton"]::before,
[data-testid="stSidebarCollapseButton"]::after,
header [data-testid="stSidebarCollapseButton"]::before,
header [data-testid="stSidebarCollapseButton"]::after,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]::before,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]::after {
  content: none !important;
  background: transparent !important;
  box-shadow: none !important;
}

/* kill hover / focus / active highlight on button and descendants */
[data-testid="stSidebarCollapseButton"]:hover,
[data-testid="stSidebarCollapseButton"]:focus,
[data-testid="stSidebarCollapseButton"]:active,
[data-testid="stSidebarCollapseButton"] *:hover,
[data-testid="stSidebarCollapseButton"] *:focus,
[data-testid="stSidebarCollapseButton"] *:active,
header [data-testid="stSidebarCollapseButton"]:hover,
header [data-testid="stSidebarCollapseButton"]:focus,
header [data-testid="stSidebarCollapseButton"]:active,
header [data-testid="stSidebarCollapseButton"] *:hover,
header [data-testid="stSidebarCollapseButton"] *:focus,
header [data-testid="stSidebarCollapseButton"] *:active,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]:focus,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]:active,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] *:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] *:focus,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] *:active {
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  -webkit-box-shadow: none !important;
  border: none !important;
  outline: none !important;
}

/* As a last resort, remove any explicit background color from immediate parent nodes
   (these are safe to force transparent for the collapse-button area only) */
header [data-testid="stHeader"] [role="button"],
header [data-testid="stHeader"] [role="button"] * {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}

/* Recolor the icon to white (optional; helps visibility on blue sidebar) */
[data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"],
header [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
  color: #ffffff !important;
  fill: #ffffff !important;
}


</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


LABEL_CSS = """
<style>
/* Force labels in the sidebar to show as white */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSelectbox label {
  color: #ffffff !important;
  font-weight: 600 !important;
}
</style>
"""
st.markdown(LABEL_CSS, unsafe_allow_html=True)

CLEAR_CHAT_CSS = """
<style>
/* Make ONLY the Clear chat button green */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="secondary"],
section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
div[aria-label="clear_chat_button"] > button {
  background-color: #32CD32 !important;   /* bright green */
  color: #ffffff !important;
  border: none !important;
  padding: 8px 14px !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
  box-shadow: none !important;
  transition: background 0.2s ease;
}

div[aria-label="clear_chat_button"] > button:hover {
  background-color: #28a428 !important;   /* darker green on hover */
}
</style>
"""
st.markdown(CLEAR_CHAT_CSS, unsafe_allow_html=True)

# ---- Add this AFTER st.markdown(CSS, unsafe_allow_html=True) ----
SIDEBAR_INNER_DROPDOWN_CSS = """
<style>
/* Make wrapper transparent so only the inner dropdown button is the visible horizontal card */
section[data-testid="stSidebar"] .stSelectbox > div {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
  padding: 0 !important;
  margin: 0px !important;
}

/* Style only the interactive inner button as a single horizontal white card */
section[data-testid="stSidebar"] .stSelectbox > div > div[role="button"],
section[data-testid="stSidebar"] .stSelectbox > div > div[role="button"] > div {
  background: #ffffff !important;
  padding: 10px 14px !important;
  border-radius: 10px !important;
  border: 1px solid rgba(0,0,0,0.06) !important;
  box-shadow: 0 6px 16px rgba(0,0,0,0.05) !important;
  width: calc(100% - 20px) !important;
  margin: 10px 10px !important;
  min-height: 46px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  box-sizing: border-box !important;
}

/* Keep the caret visible and clickable */
section[data-testid="stSidebar"] .stSelectbox svg,
section[data-testid="stSidebar"] .stSelectbox button {
  right: 12px !important;
  z-index: 9999 !important;
  fill: #222 !important;
  color: #222 !important;
}

/* Hide external label (so only inner card appears). If you want the label, set to 'block' */
section[data-testid="stSidebar"] label {
  display: none !important;
}

/* small responsive tweak */
@media (max-width: 1200px) {
  section[data-testid="stSidebar"] .stSelectbox > div > div[role="button"] {
    width: calc(100% - 16px) !important;
    margin: 8px 8px !important;
  }
}

/* Add uniform vertical spacing between sidebar controls (dropdowns, number inputs, sliders, buttons, etc.) */
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stCheckbox,
section[data-testid="stSidebar"] .stButton,
section[data-testid="stSidebar"] .stTextArea,
section[data-testid="stSidebar"] .stTextInput,
section[data-testid="stSidebar"] .stFileUploader,
section[data-testid="stSidebar"] .stDownloadButton {
  margin-bottom: 12px !important;
}



</style>
"""
st.markdown(SIDEBAR_INNER_DROPDOWN_CSS, unsafe_allow_html=True)





# ---------- Utilities ----------
def strip_html_tags(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r'(?i)<br\s*/?>', '\n', text)
    try:
        text = html_lib.unescape(text)
    except Exception:
        pass
    text = re.sub(r'<[^>]+>', '', text)
    return text

def safe_store_text(text: str) -> str:
    if text is None:
        return ""
    return strip_html_tags(text)

def df_from_sources_list(sources_list):
    try:
        if not sources_list:
            return pd.DataFrame()
        return pd.DataFrame(sources_list)
    except Exception:
        return pd.DataFrame()

# ---------- Vertex init (LLM-as-judge) ----------
# configure these to your project / region / desired model
_VERTEX_PROJECT = PROJECT
_VERTEX_LOCATION = "us-central1"            # change if needed
_VERTEX_MODEL_NAME = "gemini-2.5-pro"     # change to an available model for your project

# initialize Vertex once (safe: wrapped in try so app still runs if init fails)
try:
    vertexai.init(project=_VERTEX_PROJECT, location=_VERTEX_LOCATION)
    _MODEL_HANDLE = generative_models.GenerativeModel(_VERTEX_MODEL_NAME)
except Exception as e:
    _MODEL_HANDLE = None
    print("Vertex init failed (judge disabled):", e)
    print("MODEL HANDLE methods:", [m for m in dir(_MODEL_HANDLE) if not m.startswith("_")])

# ---------- end Vertex init ----------

# ---------- LLM-as-judge helper (use Vertex generative model) ----------
def _find_first_json_object(s: str):
    """Return the substring containing the first balanced JSON object in s, or None."""
    if not s:
        return None
    start = s.find('{')
    if start == -1:
        return None
    stack = []
    for i in range(start, len(s)):
        ch = s[i]
        if ch == '{':
            stack.append('{')
        elif ch == '}':
            if not stack:
                continue
            stack.pop()
            if not stack:
                return s[start:i+1]
    return None

def _sanitize_markdown(raw: str) -> str:
    """Remove leading/trailing markdown code fences and language markers."""
    if not raw:
        return raw
    text = raw.strip()
    text = re.sub(r"^\s*```(?:json|json\n)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip("` \n\r\t")
    return text

def _extract_percent_from_text(s: str):
    """
    Try multiple regex patterns to infer a relevance percentage from free text.
    Returns float 0..100 or None.
    """
    if not s:
        return None

    # 1) Look for explicit percent values near the words 'relevance' or 'score'
    patterns = [
        r"(?:relevance|relevance_pct|relevance%|relevance score|score)\s*[:is\-]*\s*([0-9]{1,3}(?:\.[0-9]+)?\s*%?)",
        r"(?:relevance|score)\s*\=\s*([0-9]{1,3}(?:\.[0-9]+)?\s*%?)",
        r"([0-9]{1,3}(?:\.[0-9]+)?)\s*percent",                       # e.g., "87 percent"
        r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%",                            # e.g., "87%"
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            try:
                token = m.group(1)
                token = token.strip().rstrip('%').strip()
                val = float(token)
                # If value is between 0 and 1, assume fraction
                if 0.0 <= val <= 1.0:
                    val = val * 100.0
                val = round(max(min(val, 100.0), 0.0), 2)
                return val
            except Exception:
                continue

    # 2) Heuristic: find any percent anywhere if the above failed (pick first)
    m_anypct = re.search(r"([0-9]{1,3}(?:\.[0-9]+)?)\s*%", s)
    if m_anypct:
        try:
            val = float(m_anypct.group(1))
            val = round(max(min(val, 100.0), 0.0), 2)
            return val
        except Exception:
            pass

    # 3) Look for isolated numeric tokens close to the word 'relevance' (within ~40 chars)
    m_rel = re.search(r"(relevance|score).{0,40}?([0-9]{1,3}(?:\.[0-9]+)?)", s, flags=re.IGNORECASE)
    if m_rel:
        try:
            val = float(m_rel.group(2))
            if 0.0 <= val <= 1.0:
                val = val * 100.0
            val = round(max(min(val, 100.0), 0.0), 2)
            return val
        except Exception:
            pass

    # 4) Last-resort: if the text says things like "highly relevant" / "low relevance" we can map to coarse bins (optional)
    keywords = {
        r"\b(highly relevant|very relevant|excellent match|perfect match)\b": 95.0,
        r"\b(moderately relevant|somewhat relevant|good match)\b": 70.0,
        r"\b(slightly relevant|weak match|partial match)\b": 40.0,
        r"\b(not relevant|irrelevant|no match)\b": 5.0,
    }
    for pat, score in keywords.items():
        if re.search(pat, s, flags=re.IGNORECASE):
            return score

    return None

def llm_judge_relevance(query: str, answer: str, max_retries: int = 1, allow_coerce: bool = True):
    """
    Returns (relevance_pct: float 0..100 or None, explanation: str or None, raw: str)
    - allow_coerce: if True, will make a secondary model call asking explicitly for JSON if no numeric relevance found.
    """
    global _MODEL_HANDLE
    if _MODEL_HANDLE is None:
        return None, "Vertex model not initialized", None

    # Strong instruction: ask for raw JSON only + fallback JSON if impossible
    prompt = (
        "You are a precise evaluator. Given USER QUERY and ANSWER, return ONLY valid JSON with keys:\n"
        '{"relevance": <number between 0 and 100>, "explanation": "<one-sentence reason>"}\n\n'
        "IMPORTANT: Return raw JSON only (no surrounding backticks, no markdown, no commentary). "
        "If you cannot produce valid JSON, return {\"relevance\": null, \"explanation\": \"<one-line reason>\"}.\n\n"
        "USER QUERY:\n" + json.dumps(query) + "\n\n"
        "ANSWER:\n" + json.dumps(answer) + "\n\n"
    )

    # extractor for many response shapes
    def extract_text(resp):
        if resp is None:
            return ""
        if hasattr(resp, "text") and resp.text:
            return str(resp.text).strip()
        try:
            cand = getattr(resp, "candidates", None)
            if cand:
                parts_texts = []
                for c in cand:
                    cont = getattr(c, "content", None)
                    if cont and getattr(cont, "parts", None):
                        for p in cont.parts:
                            if getattr(p, "text", None):
                                parts_texts.append(p.text)
                    elif getattr(c, "text", None):
                        parts_texts.append(c.text)
                if parts_texts:
                    return "".join(parts_texts).strip()
        except Exception:
            pass
        try:
            if hasattr(resp, "output") and resp.output:
                return str(resp.output).strip()
        except Exception:
            pass
        try:
            return str(resp).strip()
        except Exception:
            return ""

    # detect supported kwargs for generate_content
    try:
        sig = inspect.signature(_MODEL_HANDLE.generate_content)
        sig_params = set(sig.parameters.keys())
    except Exception:
        sig_params = set()
    preferred_kwargs = {
        "temperature": 0.0,
        "max_output_tokens": 200,
        "max_output_chars": 20000,
        "candidate_count": 1,
    }
    supported_kwargs = {k: v for k, v in preferred_kwargs.items() if k in sig_params}

    raw_text = ""
    for attempt in range(max_retries + 1):
        try:
            if supported_kwargs:
                resp = _MODEL_HANDLE.generate_content(prompt, **supported_kwargs)
            else:
                resp = _MODEL_HANDLE.generate_content(prompt)
            raw_text = extract_text(resp)
            break
        except TypeError:
            # fallback to calling without kwargs
            try:
                resp = _MODEL_HANDLE.generate_content(prompt)
                raw_text = extract_text(resp)
                break
            except Exception as e:
                raw_text = f"vertex call failed: {e}"
                if attempt < max_retries:
                    time.sleep(0.3)
                    continue
                else:
                    return None, raw_text, raw_text
        except Exception as e:
            raw_text = f"vertex call failed: {e}"
            if attempt < max_retries:
                time.sleep(0.3)
                continue
            else:
                return None, raw_text, raw_text

    cleaned = _sanitize_markdown(raw_text)

    # 1) Try direct JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "relevance" in parsed:
            rel = parsed["relevance"]
            if isinstance(rel, (int, float)) and 0.0 <= rel <= 1.0:
                rel = float(rel) * 100.0
            rel = round(max(min(float(rel), 100.0), 0.0), 2) if rel is not None else None
            return rel, parsed.get("explanation"), raw_text
    except Exception:
        pass

    # 2) Find first JSON object anywhere in cleaned text
    candidate_json = _find_first_json_object(cleaned)
    if candidate_json:
        try:
            parsed = json.loads(candidate_json)
            if isinstance(parsed, dict) and "relevance" in parsed:
                rel = parsed["relevance"]
                if isinstance(rel, (int, float)) and 0.0 <= rel <= 1.0:
                    rel = float(rel) * 100.0
                rel = round(max(min(float(rel), 100.0), 0.0), 2) if rel is not None else None
                return rel, parsed.get("explanation"), raw_text
        except Exception:
            pass

    # 3) Try to extract percent/numeric heuristically from the free text
    pct = _extract_percent_from_text(cleaned or raw_text)
    if pct is not None:
        # attempt to find a short explanation: the first sentence of the model output
        expl = None
        first_line = (cleaned or raw_text).splitlines()[0].strip()
        if first_line and len(first_line) < 400:
            expl = first_line
        return pct, expl, raw_text

    # 4) If allowed, attempt a coercion: ask model to OUTPUT ONLY the JSON (secondary call)
    if allow_coerce:
        coercion_prompt = (
            "You previously returned text that did not contain a numeric relevance. "
            "Now: OUTPUT ONLY valid JSON with keys {\"relevance\": <number 0..100>, \"explanation\": \"one-sentence reason\"}. "
            "If you cannot produce a numeric relevance, set relevance to null. "
            "Do not include any backticks or extra text.\n\n"
            "PREVIOUS MODEL OUTPUT:\n" + json.dumps(cleaned or raw_text) + "\n\n"
            "Return only the JSON."
        )
        try:
            if supported_kwargs:
                resp2 = _MODEL_HANDLE.generate_content(coercion_prompt, **supported_kwargs)
            else:
                resp2 = _MODEL_HANDLE.generate_content(coercion_prompt)
            raw2 = extract_text(resp2)
            raw2_clean = _sanitize_markdown(raw2)
            # try parse
            try:
                parsed2 = json.loads(raw2_clean)
                rel2 = parsed2.get("relevance", None)
                if isinstance(rel2, (int, float)) and 0.0 <= rel2 <= 1.0:
                    rel2 = float(rel2) * 100.0
                rel2 = round(max(min(float(rel2), 100.0), 0.0), 2) if rel2 is not None else None
                return rel2, parsed2.get("explanation"), raw_text + "\n\n(secondary_coercion):\n" + raw2
            except Exception:
                # if not JSON, try regex extraction on raw2
                p2 = _extract_percent_from_text(raw2_clean)
                if p2 is not None:
                    return p2, None, raw_text + "\n\n(secondary_coercion):\n" + raw2
        except Exception:
            # ignore coercion failures and fall through
            pass

    # 5) No numeric found — return None plus short explanation (first sentence of the model output)
    fallback_expl = None
    if cleaned:
        # first sentence as explanation
        sentences = re.split(r'(?<=[.!?])\s+', cleaned.strip())
        fallback_expl = sentences[0] if sentences else cleaned.strip()
        if len(fallback_expl) > 400:
            fallback_expl = fallback_expl[:400] + "..."
    elif raw_text:
        fallback_expl = (raw_text[:300] + "...") if len(raw_text) > 300 else raw_text

    return None, fallback_expl, raw_text




# ---------- Session defaults ----------
if "top_k_val" not in st.session_state:
    st.session_state.top_k_val = 5
if "max_output_tokens_val" not in st.session_state:
    st.session_state.max_output_tokens_val = 800
if "temperature_val" not in st.session_state:
    st.session_state.temperature_val = 0.2

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processing_submission" not in st.session_state:
    st.session_state.processing_submission = False

# sanitize old messages in session_state (convert legacy DataFrame to list-of-dicts)
sanitized = []
for entry in st.session_state.chat_history:
    safe_text = safe_store_text(entry.get("text", ""))
    sources = entry.get("sources", None)
    if hasattr(sources, "to_dict"):
        try:
            sources = sources.reset_index(drop=True).to_dict(orient="records")
        except Exception:
            sources = None
    sanitized.append({
        "role": entry.get("role", "user"),
        "text": safe_text,
        "time": entry.get("time", ""),
        "sources": sources,
        "meta": entry.get("meta", {})
    })
st.session_state.chat_history = sanitized

# ---------- Sidebar (Advanced settings using dropdowns) ----------
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:white;'>Solution Scope</h2> <br>", unsafe_allow_html=True)

        # Options
    ABOUT_OPTS = ["Home","About Us"]
    APPLICATION_OPTS = ["Application","Patent Q&A (RAG)", "Patent Summarization", "Similarity Search"]
    LIBRARIES_OPTS = ["Libraries","Streamlit", "google-cloud-bigquery", "pandas", "faiss", "transformers"]

    # Render in sidebar using st.sidebar.selectbox (guaranteed to be in the sidebar)
    about = st.sidebar.selectbox("Home", ABOUT_OPTS, index=0, key="about_us_sel")
    app = st.sidebar.selectbox("Application", APPLICATION_OPTS, index=0, key="application_sel")
    libs = st.sidebar.selectbox("Libraries", LIBRARIES_OPTS, index=0, key="libraries_sel")

    st.markdown(
        f"<div style='color:#ffffff; font-weight:400;'>top_k (retrieval): <span style='font-weight:400'>{st.session_state.get('top_k_val', 5)}</span></div>",
        unsafe_allow_html=True
    )

    st.session_state.top_k_val = st.number_input(
        "top_k (retrieval)", min_value=1, max_value=50,
        value=st.session_state.get("top_k_val", 5), step=1,
        format="%d", key="top_k_number"
    )

    st.markdown(
    f"<div style='color:#ffffff; font-weight:400;'>temperature: <span style='font-weight:400'>{st.session_state.get('temperature_val', 0.2)}</span></div>",
    unsafe_allow_html=True
    )
    
    st.session_state.temperature_val = st.number_input(
        "temperature", min_value=0.0, max_value=1.0,
        value=st.session_state.get("temperature_val", 0.2),
        step=0.01, format="%.2f", key="temperature_number"
    )


    st.markdown(
    f"<div style='color:#ffffff; font-weight:400;'>max_output_tokens: <span style='font-weight:400'>{st.session_state.get('max_output_tokens_val', 800)}</span></div>",
    unsafe_allow_html=True
    )
    
    st.session_state.max_output_tokens_val = st.number_input(
        "max_output_tokens", min_value=64, max_value=2000,
        value=st.session_state.get("max_output_tokens_val", 800),
        step=64, format="%d", key="max_out_number"
    )

    show_sources = st.checkbox(
        "Show top-k sources",
        value=st.session_state.get("show_sources_box", True),
        key="show_sources_box"
    )

    st.markdown("---")

    clear_chat = st.button("Clear / Reset", key="clear_chat_button", use_container_width=True)
    if clear_chat:
        st.session_state.chat_history = []
        st.success("Chat cleared.")

    # ---------------- Logos section ----------------
    import base64

    def render_logo(path, width=60):
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{data}" style="width:{width}px;"/>'

    logos_html = f"""
    <div style="text-align:center; margin-top:15px; margin-bottom:8px; font-weight:500; font-size:18px; color:white;">
        Build & Deployed on
    </div>
    <div style="display:flex; justify-content:center; align-items:center; gap:40px; margin-top:10px;">
        {render_logo("llmattscaleai.png")}
        {render_logo("gcplogo.png")}
        {render_logo("github.png")}
    </div>
    """

    st.markdown(logos_html, unsafe_allow_html=True)



# ---------- BigQuery connectivity test ----------
def test_bq_connection():
    try:
        client = bigquery.Client(project=PROJECT)
        df = client.query("SELECT 1 as ok LIMIT 1").result().to_dataframe()
        return True, f"Connected to BigQuery project: {client.project}"
    except Exception as ex:
        return False, str(ex)

if RUN_BQ_TEST:
    ok, msg = test_bq_connection()
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.warning("BigQuery connection test failed: " + msg)

# ---------- BigQuery RAG ----------
def run_rag_query(q_text: str, top_k:int, temperature:float, max_output_tokens:int, show_sources:bool):
    """Return (answer:str, sources_list:list-of-dicts)."""
    try:
        client = bigquery.Client(project=PROJECT)
    except Exception as e:
        raise RuntimeError("Failed to initialize BigQuery client: " + str(e))

    try:
        sql = f"""
        DECLARE user_query STRING DEFAULT @user_query;
        WITH q AS (
          SELECT ml_generate_embedding_result AS text_embedding
          FROM ML.GENERATE_EMBEDDING(MODEL {EMB_MODEL}, (SELECT user_query AS content))
        ),
        hits AS (
          SELECT base.publication_number, base.title, SUBSTR(base.abstract,1,1200) AS abstract, distance
          FROM VECTOR_SEARCH(TABLE {EMB_TABLE}, 'text_embedding', TABLE q, top_k => {top_k}, distance_type => 'COSINE')
          ORDER BY distance
        ),
        context AS (
          SELECT STRING_AGG(CONCAT('PUB: ', publication_number, '\\nTITLE: ', title, '\\nABSTRACT: ', abstract), '\\n\\n---\\n\\n' ORDER BY distance) AS ctx
          FROM hits
        ),
        gen AS (
          SELECT ml_generate_text_result AS gen_json
          FROM ML.GENERATE_TEXT(MODEL {LLM_MODEL},
            (SELECT CONCAT('You are a precise patent analyst. Use ONLY the CONTEXT and cite PUB IDs. QUESTION: ', user_query, '\\nCONTEXT:\\n', ctx) AS prompt FROM context),
            STRUCT({temperature} AS temperature, {max_output_tokens} AS max_output_tokens)
          )
        )
        SELECT JSON_VALUE(gen_json, '$.candidates[0].content.parts[0].text') AS answer FROM gen;
        """
        job = client.query(sql, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("user_query","STRING", q_text)]
        ), location=DEFAULT_LOCATION)
        df = job.result().to_dataframe()
        answer = df.iloc[0]["answer"] if not df.empty else "(no answer)"
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError("BigQuery query failed: " + str(e) + "\n\n" + tb)

    sources_list = []
    if show_sources:
        try:
            sql_sources = f"""
            DECLARE user_query STRING DEFAULT @user_query;
        WITH q AS (
          SELECT ml_generate_embedding_result AS text_embedding
          FROM ML.GENERATE_EMBEDDING(MODEL {EMB_MODEL}, (SELECT user_query AS content))
        )
        SELECT
          base.publication_number,
          base.title,
          base.abstract,
          distance,
          ROUND(
            GREATEST(
              LEAST(
                CASE
                  WHEN distance IS NULL THEN 0
                  WHEN distance <= 1 THEN (1 - distance) * 100
                  ELSE (1 - distance / 2) * 100
                END,
                100
              ),
              0
            ),
            2
          ) AS relevance_pct
        FROM VECTOR_SEARCH(
          TABLE {EMB_TABLE},
          'text_embedding',
          TABLE q,
          top_k => {top_k},
          distance_type => 'COSINE'
        )
        ORDER BY distance;
            """
            sjob = client.query(sql_sources, job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("user_query","STRING", q_text)]
            ), location=DEFAULT_LOCATION)
            s_df = sjob.result().to_dataframe()
            if not s_df.empty:
                sources_list = s_df.reset_index(drop=True).to_dict(orient="records")
        except Exception:
            sources_list = []

    return answer, sources_list

# ---------- BigQuery embedding quick test ----------
def run_bq_embedding_test():
    try:
        client = bigquery.Client(project=PROJECT)
    except Exception as e:
        return False, f"Failed to create BigQuery client: {e}"

    test_sql = f"""
    DECLARE sample STRING DEFAULT "test embedding text";
    SELECT *
    FROM ML.GENERATE_EMBEDDING(MODEL {EMB_MODEL}, (SELECT sample AS content))
    LIMIT 1;
    """
    try:
        job = client.query(test_sql, location=DEFAULT_LOCATION)
        df = job.result().to_dataframe()
        return True, f"Embedding test succeeded. Columns: {list(df.columns)}. Row count: {len(df)}"
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"Embedding test failed: {e}\n\nTraceback:\n{tb}"

if st.session_state.get("_run_bq_test_from_ui", False):
    st.session_state._run_bq_test_from_ui = False
    ok, message = run_bq_embedding_test()
    if ok:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)

# ---------- Logo rendering (base64 inline) ----------
def render_logo_inline(path, width=260):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return False
    b64 = base64.b64encode(data).decode()
    uri = f"data:image/png;base64,{b64}"
    st.markdown(f'<div style="text-align:center; margin-top: 0px;"><img id="custom-centered-logo" src="{uri}" width="{width}" style="display:inline-block;"/></div>', unsafe_allow_html=True)
    return True

logo_paths = [
    "llmatscaleai.png"
]

logo_shown = False
for p in logo_paths:
    if render_logo_inline(p, width=260):
        logo_shown = True
        break

if not logo_shown:
    st.markdown('<div style="text-align:center; color:#bb0000; margin-top:-6px;">Logo not found — upload to /mnt/data or supply a hosted URL.</div>', unsafe_allow_html=True)

# ---------- Centered heading & description ----------
st.markdown('<h2 class="centered-h2">Patent Vision</h2>', unsafe_allow_html=True)
st.markdown(
      """
    <div style="text-align:center; margin-top:-6px; margin-bottom:16px;">
        <p style="color:#444; font-size:17px; line-height:1.5; margin-bottom:8px;">
            Ask questions about patents and get AI-powered answers.
            Simply type your query below and click <b>Send</b>.<br>
            <i>eg. Battery thermal runaway prevention in EV</i>
        </p>
        <hr style="margin-top:20px; border:none; border-top:2px solid #888; width:100%;">
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Chat rendering ----------
chat_box = st.container()
with chat_box:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.chat_history):
        role, text, ts = msg.get("role"), msg.get("text", ""), msg.get("time", "")
        meta = msg.get("meta", {})
        display_text = strip_html_tags(text)

        if role == "user":
            cols = st.columns([1, 6, 1])
            with cols[1]:
                st.markdown(f"<div class='meta' style='text-align:right'>You · {ts}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='user-bubble'>{html_lib.escape(display_text).replace('\\n','<br/>')}</div>",
                    unsafe_allow_html=True
                )
        else:
            cols = st.columns([1, 6, 1])
            with cols[0]:
                # render meta line including elapsed and judge relevance if available
                meta_line = f"Assistant · {ts}"
                if meta.get("elapsed"):
                    meta_line += f" · {meta.get('elapsed')}s"
                if meta.get("relevance_pct") is not None:
                    meta_line += f" · relevance: {meta.get('relevance_pct')}%"
                st.markdown(f"<div class='meta'>{meta_line}</div>", unsafe_allow_html=True)
            cols2 = st.columns([0.6, 9, 0.6])
            with cols2[1]:
                md_html = html_lib.escape(display_text).replace("\n","<br/>")
                st.markdown(
                    f"<div class='assistant-bubble'>{md_html}</div>",
                    unsafe_allow_html=True
                )

                # optional: show judge explanation under the answer
                if meta.get("relevance_expl"):
                    st.markdown(f"*Judge:* {html_lib.escape(str(meta.get('relevance_expl')))}")

                # Sources as expanders
                sources_list = msg.get("sources", None)
                sources_df = df_from_sources_list(sources_list)
                if sources_df is not None and not sources_df.empty:
                    for i, row in sources_df.reset_index(drop=True).iterrows():
                        pub = str(row.get("publication_number", ""))
                        title = str(row.get("title", ""))
                        dist = row.get("distance", "")
                        abstract = str(row.get("abstract", ""))
                        relevance = row.get("relevance_pct","")

                        label = f"{i+1}. {pub} — {title}"
                        with st.expander(label, expanded=False):
                            st.markdown("**Pub ID:** " + pub)
                            st.markdown("**Title:** " + title)
                            st.markdown("**Distance:** " + str(dist))
                            if relevance is not None and relevance != "":
                                st.markdown("**Relevance:** " + str(relevance) + " %")   # 👈 added line
                            if abstract:
                                snippet = abstract
                                if len(snippet) > 500:
                                    snippet = snippet[:500] + "..."
                                st.markdown("**Abstract:**")
                                st.write(snippet)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Input (Send form) ----------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Your message", value="", height=120, placeholder="Ask about patents...")
    submitted = st.form_submit_button("Send")

if submitted and user_input and user_input.strip():
    if st.session_state.processing_submission:
        st.warning("Still processing previous message — please wait.")
    else:
        st.session_state.processing_submission = True
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({
            "role": "user",
            "text": safe_store_text(user_input),
            "time": ts,
            "sources": None,
            "meta": {}
        })

        # Run retrieval + generation (with LLM-as-judge scoring)
        with st.spinner("Running retrieval + generation..."):
            try:
                start = datetime.datetime.utcnow()
                answer, sources_list = run_rag_query(
                    user_input.strip(),
                    top_k=st.session_state.top_k_val,
                    temperature=st.session_state.temperature_val,
                    max_output_tokens=st.session_state.max_output_tokens_val,
                    show_sources=st.session_state.get("show_sources_box", True)
                )
                elapsed = round((datetime.datetime.utcnow() - start).total_seconds(), 2)

                # --- Call LLM judge to get overall relevance ---
                try:
                    relevance_val, relevance_expl, relevance_raw = llm_judge_relevance(user_input.strip(), answer)
                except Exception as ex_j:
                    relevance_val, relevance_expl, relevance_raw = None, f"judge error: {ex_j}", None

                # append assistant message including relevance_pct in meta
                st.session_state.chat_history.append({
                    "role":"assistant",
                    "text": safe_store_text(answer or "(no answer)"),
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sources": sources_list if (st.session_state.get("show_sources_box", True) and sources_list) else None,
                    "meta": {"elapsed": elapsed, "relevance_pct": relevance_val, "relevance_expl": relevance_expl}
                })
            except Exception as e:
                st.exception(e)
                st.session_state.chat_history.append({
                    "role":"assistant",
                    "text": safe_store_text("Error: " + str(e)),
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sources": None,
                    "meta": {}
                })
            finally:
                st.session_state.processing_submission = False

        # try modern rerun, fallback safely
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

# ---------- Downloads ----------
last_assistant = None
for m in reversed(st.session_state.chat_history):
    if m.get("role") == "assistant":
        last_assistant = m
        break
if last_assistant:
    st.markdown("<hr/>", unsafe_allow_html=True)
    cols = st.columns([1,1,4])
    with cols[0]:
        st.download_button("⬇️ Answer", data=html_lib.unescape(last_assistant["text"]), file_name="rag_answer.txt")
    with cols[1]:
        payload = json.dumps({"answer": html_lib.unescape(last_assistant["text"])}, indent=2)
        st.download_button("💾 JSON", data=payload, file_name="rag_answer.json")
    with cols[2]:
        if last_assistant.get("sources") is not None:
            with st.expander("View all sources (table)"):
                s_df = df_from_sources_list(last_assistant.get("sources"))
                if not s_df.empty:
                    st.dataframe(s_df)
                else:
                    st.write("No sources to show.")

st.caption("If you see ADC errors when calling BigQuery, run `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS.")

