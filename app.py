import os
import json
import glob
import sqlite3
import datetime
import re
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client

# Optional semantic search (TF-IDF). If scikit-learn isn't available, code falls back to token-overlap.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------- CONFIG ----------
load_dotenv()
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

SCHEMA_FOLDER = os.getenv("SCHEMA_FOLDER", "./schemas")
DB_FILE = os.getenv("DB_FILE", "chat_history.db")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemini-3-flash-preview:latest")
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY")

# ---------- GLOBAL STORES ----------
GLOBAL_SCHEMA_STORE = {}        # name -> full content (string)
GLOBAL_SCHEMA_SUMMARIES = {}   # name -> short summary used for vector search
GLOBAL_ALL_NAMES = []
VECTOR_INDEX = None            # TF-IDF vectorizer
VECTOR_DOCS = []               # documents used in TF-IDF

# ---------- DB helpers ----------
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, title TEXT, created_at DATETIME)''')
        c.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at DATETIME)''')
        conn.commit()
        conn.close()
        logging.info("Database initialized.")
    except Exception as e:
        logging.exception("Database init error")


def get_chat_history_formatted(session_id, limit=10):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?", (session_id, limit))
    rows = c.fetchall()
    conn.close()
    history = []
    for r in rows[::-1]:
        history.append({"role": r["role"], "content": r["content"]})
    return history


def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
              (session_id, role, content, datetime.datetime.now()))
    conn.commit()
    conn.close()


def create_session_if_not_exists(session_id, first_msg):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if not c.fetchone():
        c.execute("INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
                  (session_id, first_msg[:50], datetime.datetime.now()))
        conn.commit()
    conn.close()

# ---------- Schema parsing utilities ----------

def _normalize_name(name: str) -> str:
    return name.strip()


def extract_columns_from_ddl(ddl: str):
    """Try to extract column names from a CREATE TABLE DDL-ish string.
    This is heuristic but helps the model not invent columns.
    """
    cols = []
    try:
        # find the first parenthesis block which usually contains column defs
        m = re.search(r"\((.*)\)\s*(?:;|$)", ddl, flags=re.DOTALL)
        if not m:
            # fallback: take lines that look like `name type` pairs
            lines = ddl.splitlines()
        else:
            block = m.group(1)
            lines = block.splitlines()

        for line in lines:
            # remove trailing commas and comments
            line = re.sub(r"--.*$|/\*.*?\*/", "", line).strip()
            line = line.rstrip(',').strip()
            if not line:
                continue
            # column patterns: name type [constraints]
            parts = re.split(r"\s+", line)
            if len(parts) >= 2:
                col = parts[0].strip('`"')
                # avoid lines that start with constraint/primary/unique
                if re.match(r"^(constraint|primary|unique|foreign|index|check)$", col, flags=re.I):
                    continue
                cols.append(col)
    except Exception:
        logging.exception("extract_columns_from_ddl failed")
    return list(dict.fromkeys(cols))  # unique preserve order


def extract_case_mappings(routine_def: str):
    """Extract WHEN ... THEN ... mapping pairs from routine SQL (heuristic)."""
    mappings = []
    try:
        # capture forms like WHEN status_id = 2 THEN 'New' or WHEN status = 'active' THEN 1
        for m in re.finditer(r"WHEN\s+([^T]+?)\s+THEN\s+('(?:[^']*)'|\d+|\"(?:[^\"]*)\")",
                             routine_def, flags=re.IGNORECASE | re.DOTALL):
            cond = m.group(1).strip()
            out = m.group(2).strip().strip("'\"")
            mappings.append({"when": cond, "then": out})
    except Exception:
        logging.exception("extract_case_mappings failed")
    return mappings


def chunk_text(text, max_size=4000):
    """Chunk text into pieces not exceeding max_size chars by splitting on newlines.
    Keeps chunks to logical blocks where possible.
    """
    if not text:
        return []
    if len(text) <= max_size:
        return [text]
    lines = text.splitlines(True)
    chunks = []
    cur = ""
    for ln in lines:
        if len(cur) + len(ln) > max_size and cur:
            chunks.append(cur)
            cur = ln
        else:
            cur += ln
    if cur:
        chunks.append(cur)
    return chunks

# ---------- Loading schemas (improved) ----------

def load_all_schemas():
    """Load schemata from JSON files under SCHEMA_FOLDER.
    Build: GLOBAL_SCHEMA_STORE (name -> full content) and GLOBAL_SCHEMA_SUMMARIES (name -> short text).
    Also build a TF-IDF index if sklearn is available.
    """
    global GLOBAL_SCHEMA_STORE, GLOBAL_SCHEMA_SUMMARIES, GLOBAL_ALL_NAMES, VECTOR_INDEX, VECTOR_DOCS
    GLOBAL_SCHEMA_STORE = {}
    GLOBAL_SCHEMA_SUMMARIES = {}
    GLOBAL_ALL_NAMES = []
    VECTOR_INDEX = None
    VECTOR_DOCS = []

    logging.info("Loading schema files from %s", SCHEMA_FOLDER)
    if not os.path.exists(SCHEMA_FOLDER):
        logging.warning("Schema folder not found: %s", SCHEMA_FOLDER)
        return

    json_files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))
    for fpath in json_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    name = None
                    full_content = None
                    summary = None

                    # Normalize multiple possible fields
                    if 'table_name' in item or 'name' in item and 'ddl' in item:
                        name = item.get('table_name') or item.get('name')
                        ddl = item.get('ddl', '')
                        cols = extract_columns_from_ddl(ddl)
                        full_content = f"[TABLE]\nName: {name}\nDDL:\n{ddl}\nColumns: {cols}\n"
                        summary = f"TABLE {name} columns: {', '.join(cols[:10])}"

                    elif 'routine_name' in item or 'routine' in item:
                        name = item.get('routine_name') or item.get('routine')
                        routine_def = item.get('routine_definition', '')
                        ddl = item.get('ddl', '')
                        args = item.get('arguments', [])
                        mappings = extract_case_mappings(routine_def)
                        full_content = (
                            f"[ROUTINE]\nName: {name}\nArguments: {json.dumps(args, ensure_ascii=False)}\n"
                            f"Definition:\n{routine_def}\nDDL:\n{ddl}\nMappings: {mappings}\n"
                        )
                        summary = f"ROUTINE {name} args: {args} mappings: {len(mappings)}"

                    # Allow other structured possibilities
                    elif 'name' in item and 'definition' in item:
                        name = item.get('name')
                        full_content = json.dumps(item, ensure_ascii=False, indent=2)
                        summary = f"OBJECT {name}"

                    if name and full_content:
                        name = _normalize_name(name)
                        # store chunked version if large
                        GLOBAL_SCHEMA_STORE[name] = full_content
                        GLOBAL_SCHEMA_SUMMARIES[name] = summary
                        GLOBAL_ALL_NAMES.append(name)
                        VECTOR_DOCS.append(summary + '\n' + full_content[:2000])
        except Exception:
            logging.exception("Failed to load schema file %s", fpath)

    # Build TF-IDF index if available
    if SKLEARN_AVAILABLE and VECTOR_DOCS:
        try:
            VECTOR_INDEX = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=20000)
            VECTOR_INDEX.fit(VECTOR_DOCS)
            logging.info("TF-IDF vector index built with %d docs", len(VECTOR_DOCS))
        except Exception:
            logging.exception("Failed to build TF-IDF index")
            VECTOR_INDEX = None

    logging.info("Loaded %d schema objects", len(GLOBAL_ALL_NAMES))

# ---------- Semantic / hybrid selection ----------

def _tfidf_rank(query: str, top_k=5):
    """Return top_k names by TF-IDF cosine similarity. Fallback to token overlap if TF-IDF not available."""
    scores = []
    if VECTOR_INDEX is None or not SKLEARN_AVAILABLE:
        # token overlap heuristic
        q_tokens = set(re.findall(r"\w+", query.lower()))
        for name, summary in GLOBAL_SCHEMA_SUMMARIES.items():
            doc_tokens = set(re.findall(r"\w+", (summary + ' ' + GLOBAL_SCHEMA_STORE.get(name, '')).lower()))
            overlap = len(q_tokens & doc_tokens)
            scores.append((name, overlap))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [n for n, s in scores[:top_k] if s > 0]

    # TF-IDF similarity
    try:
        docs = VECTOR_DOCS
        vecs = VECTOR_INDEX.transform(docs)
        qv = VECTOR_INDEX.transform([query])
        import numpy as np
        cos = (vecs @ qv.T).toarray().ravel()
        ranked = sorted(enumerate(cos), key=lambda x: x[1], reverse=True)
        result = []
        for idx, sc in ranked[:top_k]:
            # map back to name: we ensured VECTOR_DOCS order follows GLOBAL_ALL_NAMES order
            if idx < len(GLOBAL_ALL_NAMES):
                name = GLOBAL_ALL_NAMES[idx]
                result.append(name)
        return result
    except Exception:
        logging.exception("tfidf rank failed")
        return []


def ai_select_relevant_schemas(client: Client, user_msg: str, top_k=6):
    """Hybrid selection: ask LLM for a JSON list AND use TF-IDF/token overlap to rank and union results.
    This reduces LLM hallucination of names and increases recall.
    """
    proposals = []
    if GLOBAL_ALL_NAMES:
        # Request LLM to return JSON array of names only to make parsing robust
        selection_prompt = (
            "You are a data assistant. Given the user's question and the following brief index of available TABLEs and ROUTINEs, "
            "return a JSON array (e.g. [\"table_a\", \"routine_b\"]) of the names that are most useful to answer the user's question. "
            "ONLY return a JSON array. DO NOT add any extra commentary.\n\n"
            f"AVAILABLE OBJECTS:\n" + "\n".join([f"- {n}: {GLOBAL_SCHEMA_SUMMARIES.get(n,'')}" for n in GLOBAL_ALL_NAMES[:200]]) + "\n\n"
            f"USER QUESTION: {user_msg}"
        )
        try:
            resp = client.chat(model=MODEL_NAME,
                               messages=[{"role": "user", "content": selection_prompt}],
                               stream=False,
                               options={"temperature": 0.0})
            raw = resp.get('message', {}).get('content', '')
            # Try parse JSON inside the response robustly
            m = re.search(r"\[.*\]", raw, flags=re.S)
            if m:
                arr = json.loads(m.group(0))
                # keep only those that match existing names
                proposals = [p for p in arr if p in GLOBAL_ALL_NAMES]
        except Exception:
            logging.exception("LLM selection failed")

    # Hybrid: also use TF-IDF/token ranking
    tfidf_top = _tfidf_rank(user_msg, top_k=top_k)

    # Union: keep order by LLM proposals first then tfidf
    final = []
    for p in (proposals + tfidf_top):
        if p not in final:
            final.append(p)
    # expand to at most top_k unique
    final = final[:top_k]
    logging.info("Selected schemas: %s", final)
    return final

# ---------- SQL generation step (improved prompt construction) ----------

def build_system_prompt(selected_names: list):
    """Compose a compact but information-rich system prompt including columns and mappings extracted.
    This prevents the model from inventing columns/fields.
    """
    parts = ["You are a BigQuery SQL expert. Use only the schemas and routines provided; do not invent tables or columns."]
    for name in selected_names:
        content = GLOBAL_SCHEMA_STORE.get(name, '')
        cols = extract_columns_from_ddl(content)
        mappings = []
        try:
            mappings = extract_case_mappings(content)
        except Exception:
            pass
        parts.append(f"OBJECT: {name}\nColumns: {cols}\nMappings: {mappings}\n")
    parts.append(
        "Rules:\n- Produce only SQL inside a ```sql ... ``` block.\n- Prefer JOINs over correlated subqueries.\n- If a routine maps a label to an ID, use the ID in filters per the routine mapping.\n- If a requested field doesn't exist in provided schemas, say you cannot answer exactly and explain which fields are missing."
    )
    return "\n".join(parts)

# ---------- Flask routes ----------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/reload', methods=['POST'])
def reload_schema():
    load_all_schemas()
    return jsonify({"status": "success", "count": len(GLOBAL_ALL_NAMES)})


@app.route('/api/select', methods=['POST'])
def api_select():
    data = request.json
    api_key = data.get('api_key') or DEFAULT_API_KEY
    user_msg = data.get('message')
    if not api_key:
        return jsonify({"error": "Missing api_key"}), 401
    if not user_msg:
        return jsonify({"error": "Missing message"}), 400
    client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
    sel = ai_select_relevant_schemas(client, user_msg)
    return jsonify({"selected": sel})


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    api_key = data.get('api_key') or DEFAULT_API_KEY
    user_msg = data.get('message')
    session_id = data.get('session_id')
    if not api_key:
        return jsonify({"error": "Missing api_key"}), 401
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})

    create_session_if_not_exists(session_id, user_msg)
    save_message(session_id, "user", user_msg)

    # Stage 1: select relevant schemas (hybrid)
    selected = ai_select_relevant_schemas(client, user_msg, top_k=8)

    # Stage 2: load full context (chunks) with careful size limit
    context_chunks = []
    MAX_CONTEXT_CHARS = 120000
    cur_len = 0
    for name in selected:
        content = GLOBAL_SCHEMA_STORE.get(name, '')
        for chunk in chunk_text(content, max_size=6000):
            if cur_len + len(chunk) <= MAX_CONTEXT_CHARS:
                context_chunks.append(f"-- {name}\n" + chunk)
                cur_len += len(chunk)

    final_context = "\n\n".join(context_chunks)

    # Build system prompt and messages
    system_prompt = build_system_prompt(selected)
    system_prompt = system_prompt + "\n\nCONTEXT:\n" + final_context

    messages_payload = [{"role": "system", "content": system_prompt}]
    # include some recent history (safely truncated)
    history = get_chat_history_formatted(session_id, limit=6)
    for msg in history:
        if msg['content'] != user_msg:
            messages_payload.append(msg)
    messages_payload.append({"role": "user", "content": user_msg})

    try:
        resp = client.chat(model=MODEL_NAME, messages=messages_payload, stream=False, options={"temperature": 0.05})
        ai_reply = resp.get('message', {}).get('content', '')
        save_message(session_id, "assistant", ai_reply)
        return jsonify({"response": ai_reply, "selected": selected})
    except Exception as e:
        logging.exception("chat error")
        return jsonify({"error": str(e)}), 500


# ---------- startup ----------
if __name__ == '__main__':
    init_db()
    load_all_schemas()
    app.run(debug=True, port=int(os.getenv('PORT', 5000)))
