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


# appp_chat_fixed.py
# Patent RAG — Chat (fixed)
# Run: streamlit run appp_chat_fixed.py

import streamlit as st
from google.cloud import bigquery
import pandas as pd
import datetime
import html as html_lib
import json
import re
import traceback

# ---------- CONFIG ----------
# Set your GCP project id here
PROJECT = "genai-poc-424806"  # <<--- set this
DATASET = "patent_demo"
EMB_MODEL = f"`{PROJECT}.{DATASET}.embedding_model`"
LLM_MODEL = f"`{PROJECT}.{DATASET}.gemini_text`"
EMB_TABLE = f"`{PROJECT}.{DATASET}.patent_embeddings`"
DEFAULT_LOCATION = "US"

# When True, run a small BQ connectivity test on startup (helpful while debugging)
RUN_BQ_TEST = True

st.set_page_config(page_title="Patent RAG — Chat", layout="wide", initial_sidebar_state="expanded")

# ---------- CSS / THEME ----------
CSS = """
<style>
/* ---------- FORCEFUL THEME: copy-paste into appp.py ---------- */

/* 1) Sidebar: very specific + fallback */
section[data-testid="stSidebar"],
div[data-testid="stSidebar"] {
  background-color: #318CE7 !important;
  padding: 3rem 1.5rem !important;
  color: #ffffff !important;
}

/* fallback: exact classes you provided (kept) */
.st-emotion-cache-6qob1r,
#root > div:nth-child(1) > div.withScreencast > div > div > div > section.st-emotion-cache-vk3wp9.eczjsme11 > div.st-emotion-cache-6qob1r.eczjsme3 > div.st-emotion-cache-16txtl3.eczjsme4 {
  background-color: #318CE7 !important;
}

/* 2) Header height */
header[data-testid="stHeader"] { height: 57px !important; }

/* 3) Buttons inside sidebar - force green */
section[data-testid="stSidebar"] form input[type="submit"],
section[data-testid="stSidebar"] button {
  background-color: #32CD32 !important;
  color: white !important;
  border: none !important;
}

/* 4) Main page logo fallback (keeps your placement) */
#root img[alt="logo"], #root > div img {
  max-width: 65% !important;
  margin-left: 25% !important;
  position: absolute !important;
  top: -100px !important;
}

/* 5) Chat area / textareas */
.stTextArea textarea,
.stTextArea > div > textarea {
  background: #318CE7 !important;
  color: #ffffff !important;
  border-radius: 8px !important;
}

/* 6) Chat bubbles classes (use them in HTML) */
.user-bubble {
  background: #f6f8fa !important;
  color: #05386b !important;
  padding: 12px !important;
  border-radius: 12px !important;
  white-space: pre-wrap !important;
}
.assistant-bubble {
  background: linear-gradient(180deg,#05386b,#318CE7) !important;
  color: #ffffff !important;
  padding: 12px !important;
  border-radius: 12px !important;
  white-space: pre-wrap !important;
}

/* 7) Hide footer */
footer { visibility: hidden !important; }

/* 8) Last-resort override: target large container elements to force sidebar color */
#root > div > div > div > section[data-testid="stSidebar"] {
  background-color: #318CE7 !important;
}

/* responsive */
@media (max-width: 800px) {
  section[data-testid="stSidebar"] { padding: 1.25rem !important; }
}
</style>
    """
st.markdown(CSS, unsafe_allow_html=True)

EXTRA_CSS = """
<style>
/* 1. Advanced settings expander */
.streamlit-expanderHeader,
div[aria-expanded="true"].streamlit-expanderHeader {
  background-color: #318CE7 !important;
  color: white !important;
  border-radius: 5px !important;
  font-weight: 600 !important;
}

/* 2. Centered title */
h2, h1, .stMarkdown h2 {
  text-align: center !important;
  color: #318CE7 !important;
  font-size: 32px !important;
  font-weight: 700 !important;
}

/* 3. Assistant chat bubble */
.assistant-bubble {
  background: #318CE7 !important;
  color: #ffffff !important;
  padding: 12px !important;
  border-radius: 12px !important;
  white-space: pre-wrap !important;
}
</style>
"""
st.markdown(EXTRA_CSS, unsafe_allow_html=True)

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
    """Reconstruct DataFrame from a list-of-dicts or return empty df."""
    try:
        if not sources_list:
            return pd.DataFrame()
        return pd.DataFrame(sources_list)
    except Exception:
        return pd.DataFrame()

# ---------- Session state ----------
if "chat_history" not in st.session_state:
    # chat_history: list of entries with keys:
    # role (user|assistant), text (str), time (str), sources (list-of-dicts or None), meta (dict)
    st.session_state.chat_history = []

if "processing_submission" not in st.session_state:
    st.session_state.processing_submission = False

# sanitize old messages in session_state (convert legacy DataFrame to list-of-dicts)
sanitized = []
for entry in st.session_state.chat_history:
    safe_text = safe_store_text(entry.get("text", ""))
    sources = entry.get("sources", None)
    # If the stored sources looks like a DataFrame, convert to list-of-dicts
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

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Controls")
    with st.expander("Advanced settings", expanded=False):
        top_k = st.number_input("top_k (retrieval)", 1, 20, 5)
        temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
        max_output_tokens = st.number_input("max_output_tokens", 64, 2000, 800, 50)
        show_sources = st.checkbox("Show top-k sources", value=True)
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

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

    # Main generation SQL
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
        # provide traceback context
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
            SELECT base.publication_number, base.title, base.abstract, distance
            FROM VECTOR_SEARCH(TABLE {EMB_TABLE}, 'text_embedding', TABLE q, top_k => {top_k}, distance_type => 'COSINE')
            ORDER BY distance;
            """
            sjob = client.query(sql_sources, job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("user_query","STRING", q_text)]
            ), location=DEFAULT_LOCATION)
            s_df = sjob.result().to_dataframe()
            if not s_df.empty:
                # convert to list-of-dicts for safe storage in session_state
                sources_list = s_df.reset_index(drop=True).to_dict(orient="records")
        except Exception:
            # if sources fetch fails, return empty list but don't block the answer
            sources_list = []

    return answer, sources_list

# ---------- Chat rendering ----------
st.markdown("<h2 style='color:var(--text-color)'> Patent Vision</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <p style="color:gray; font-size:16px; margin-top:-10px;">
     Ask questions about patents and get AI-powered answers.  
    Simply type your query below and click <b>Send</b>.
    eg. Battery thermal runaway prevention in EV
    </p>
    """,
    unsafe_allow_html=True
)

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
                st.markdown(f"<div class='meta'>Assistant · {ts}{(' · '+str(meta.get('elapsed'))+'s') if meta.get('elapsed') else ''}</div>", unsafe_allow_html=True)
            cols2 = st.columns([0.6, 9, 0.6])
            with cols2[1]:
                md_html = html_lib.escape(display_text).replace("\n","<br/>")
                st.markdown(
                    f"<div class='assistant-bubble'>{md_html}</div>",
                    unsafe_allow_html=True
                )

                # --- Sources dropdown ---
                sources_list = msg.get("sources", None)
                sources_df = df_from_sources_list(sources_list)
                if sources_df is not None and not sources_df.empty:
                    options = []
                    for i, row in sources_df.reset_index(drop=True).iterrows():
                        pub = str(row.get("publication_number", ""))
                        title = str(row.get("title", ""))
                        options.append(f"{i+1}. {pub} — {title}")
                    select_key = f"src_select_{idx}"
                    chosen = st.selectbox("Sources (select to view details)", options, key=select_key)
                    sel_index = int(chosen.split(".")[0]) - 1
                    sel_row = sources_df.reset_index(drop=True).iloc[sel_index]
                    st.markdown("**Pub ID:** " + str(sel_row.get("publication_number", "")))
                    st.markdown("**Title:** " + str(sel_row.get("title", "")))
                    st.markdown("**Distance:** " + str(sel_row.get("distance", "")))
                    if "abstract" in sel_row.index:
                        snippet = str(sel_row.get("abstract", ""))
                        if len(snippet) > 500:
                            snippet = snippet[:500] + "..."
                        st.markdown("**Abstract:** " + snippet)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Input ----------
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Your message", value="", height=120, placeholder="Ask about patents...")
    submitted = st.form_submit_button("Send")

if submitted and user_input and user_input.strip():
    if st.session_state.processing_submission:
        st.warning("Still processing previous message — please wait.")
    else:
        # Mark processing so parallel clicks are prevented
        st.session_state.processing_submission = True
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # store user message
        st.session_state.chat_history.append({
            "role": "user",
            "text": safe_store_text(user_input),
            "time": ts,
            "sources": None,
            "meta": {}
        })

        # Run retrieval + generation
        with st.spinner("Running retrieval + generation..."):
            try:
                start = datetime.datetime.utcnow()
                answer, sources_list = run_rag_query(user_input.strip(), top_k, temperature, max_output_tokens, show_sources)
                elapsed = round((datetime.datetime.utcnow() - start).total_seconds(), 2)
                st.session_state.chat_history.append({
                    "role":"assistant",
                    "text": safe_store_text(answer or "(no answer)"),
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sources": sources_list if (show_sources and sources_list) else None,
                    "meta": {"elapsed": elapsed}
                })
            except Exception as e:
                # Show full traceback in-app for easier debugging
                st.exception(e)
                # Also append an assistant-level error so the user sees something in chat history
                st.session_state.chat_history.append({
                    "role":"assistant",
                    "text": safe_store_text("Error: " + str(e)),
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sources": None,
                    "meta": {}
                })
            finally:
                st.session_state.processing_submission = False

        # Rerun so the updated chat_history is rendered immediately
        st.rerun()

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

