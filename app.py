# full_precision_rag_sql_chat_app.py
# Precision-first RAG + BigQuery SQL generator (Flask)

import os
import json
import glob
import datetime
import re
import difflib
from collections import defaultdict

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from rank_bm25 import BM25Okapi

# =========================================================
# CONFIG
# =========================================================

load_dotenv()
app = Flask(__name__)
CORS(app)

DB_URL = os.getenv("DATABASE_URL") or "sqlite:///local_chat.db"
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

OLLAMA_HOST = os.getenv("OLLAMA_HOST") or "https://ollama.com"
MODEL_NAME = os.getenv("MODEL_NAME") or "gemini-3-flash-preview:latest"
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY")
SCHEMA_FOLDER = os.getenv("SCHEMA_FOLDER") or "./schemas"
PROJECT_ID = os.getenv("PROJECT_ID") or "kynaforkids-server-production"

# =========================================================
# DB MODELS
# =========================================================

class Session(db.Model):
    __tablename__ = 'sessions'
    id = db.Column(db.String(50), primary_key=True)
    title = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)


class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), db.ForeignKey('sessions.id', ondelete="CASCADE"), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)


def init_db():
    with app.app_context():
        db.create_all()


def save_message(session_id, role, content):
    try:
        db.session.add(Message(session_id=session_id, role=role, content=content))
        db.session.commit()
    except Exception:
        db.session.rollback()


def create_session_if_not_exists(session_id, first_msg):
    try:
        if not Session.query.get(session_id):
            title = (first_msg[:50] + '...') if len(first_msg) > 50 else first_msg
            db.session.add(Session(id=session_id, title=title))
            db.session.commit()
    except Exception:
        db.session.rollback()


def get_chat_history_formatted(session_id, limit=10):
    try:
        msgs = Message.query.filter_by(session_id=session_id) \
            .order_by(desc(Message.created_at)).limit(limit).all()
        return [{"role": m.role, "content": m.content} for m in msgs[::-1]]
    except Exception:
        return []

# =========================================================
# RAG ENGINE (precision-first)
# =========================================================

class RAGEngine:
    def __init__(self):
        self.docs = []     # raw textual docs for prompt
        self.tokens = []   # token lists for BM25
        self.meta = {}     # metadata per doc index
        self.bm25 = None
        self.ready = False

    def _normalize_identifier(self, s: str) -> str:
        if not s:
            return ""
        return re.sub(r'[`\\s]+', ' ', s).strip().lower()

    def tokenize(self, text: str):
        if not text:
            return []
        t = re.sub(r'[\-_\(\),;:\.]', ' ', text)
        t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
        t = re.sub(r'[^0-9a-zA-Z_\s]', ' ', t)
        tokens = [tok for tok in t.lower().split() if len(tok) > 2]
        stop = {'select','from','where','and','or','join','on','group','by','order','limit','as'}
        return [tok for tok in tokens if tok not in stop]

    def load_schemas(self):
        self.docs = []
        self.tokens = []
        self.meta = {}
        json_files = glob.glob(os.path.join(SCHEMA_FOLDER, '*.json'))
        for fp in json_files:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {fp}: {e}")
                continue

            items = data if isinstance(data, list) else [data]
            for item in items:
                # TABLE entries
                if 'table_name' in item:
                    table = item.get('table_name')
                    dataset = item.get('table_schema') or item.get('dataset') or ''
                    project = item.get('project') or PROJECT_ID

                    ddl = item.get('ddl','') or ''
                    m = re.search(r'`([^`]+)\.([^`]+)\.([^`]+)`', ddl)
                    if m:
                        project, dataset, table = m.groups()

                    fqtn = f"`{project}.{dataset}.{table}`" if dataset else f"`{project}.{table}`"

                    cols_raw = item.get('columns', [])
                    if isinstance(cols_raw, str):
                        try:
                            cols_raw = json.loads(cols_raw)
                        except Exception:
                            cols_raw = []
                    cols = [str(c).strip() for c in cols_raw if c]

                    doc = f"[TABLE] {fqtn}\nType: {item.get('table_type','')}\nColumns:\n" + "\n".join([f"- {c}" for c in cols])
                    kws = " ".join([fqtn, project, dataset, table] + cols + [ddl[:1000]])

                    idx = len(self.docs)
                    self.docs.append(doc)
                    tok = self.tokenize(kws)
                    self.tokens.append(tok)
                    self.meta[idx] = {
                        'type': 'table',
                        'fqtn': fqtn,
                        'project': project,
                        'dataset': dataset,
                        'table': table,
                        'columns': set([c.lower() for c in cols]),
                        'tokens_set': set(tok)
                    }

                # FUNCTION / ROUTINE entries
                elif 'routine_name' in item or 'routine_definition' in item:
                    rname = item.get('routine_name') or item.get('routine_id') or ''
                    ddl = item.get('ddl','') or ''
                    m = re.search(r'FUNCTION\s+`([^`]+)`', ddl, re.I)
                    fname = f"`{m.group(1)}`" if m else f"`{rname}`"
                    definition = item.get('routine_definition','') or item.get('definition','')

                    doc = f"[FUNCTION] {fname}\n{definition}"
                    kws = fname + " " + (definition[:1000])

                    idx = len(self.docs)
                    self.docs.append(doc)
                    tok = self.tokenize(kws)
                    self.tokens.append(tok)
                    self.meta[idx] = {
                        'type': 'function',
                        'name': fname,
                        'definition': definition,
                        'tokens_set': set(tok)
                    }

        if self.tokens:
            try:
                self.bm25 = BM25Okapi(self.tokens)
                self.ready = True
                print(f"✅ RAG indexed {len(self.docs)} docs")
            except Exception as e:
                print(f"BM25 init error: {e}")
                self.ready = False
        else:
            print("⚠️ No schema docs indexed")
            self.ready = False

    def expand_query(self, user_query, api_key):
        # conservative expansion: at most 6 semantic keywords
        try:
            client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
            prompt = (
                f"You are a keywords extractor. Given the user query, return at most 6 single-word keywords (do NOT return table names).\n"
                f"User query: {user_query}\nOutput: keywords separated by spaces."
            )
            resp = client.chat(model=MODEL_NAME, messages=[{"role":"user","content":prompt}],
                               options={"temperature":0.0, "top_p":0.8, "top_k":40})
            content = resp.get('message',{}).get('content','') or ''
            kw = re.findall(r"[A-Za-z_]+", content)
            return " ".join([w.lower() for w in kw][:6])
        except Exception:
            return ""

    def retrieve(self, query, expanded_query=None, top_k=8):
        if not self.ready:
            return [], []

        search = query + (" " + expanded_query if expanded_query else "")
        q_tokens = set(self.tokenize(search))
        if not q_tokens:
            return [], []

        bm_scores = self.bm25.get_scores(list(q_tokens)) if self.bm25 else [0]*len(self.docs)
        max_b = max(bm_scores) if bm_scores else 1.0

        scored = []
        for i, b in enumerate(bm_scores):
            meta = self.meta.get(i, {})
            if b <= 0:
                continue
            exact = len(q_tokens & meta.get('tokens_set', set()))
            exact_score = exact / max(1, len(meta.get('tokens_set', set())))
            col_hits = 0
            fuzzy = 0
            if meta.get('type') == 'table':
                col_hits = len([t for t in q_tokens if t in meta.get('columns', set())])
                if col_hits == 0:
                    for qt in q_tokens:
                        close = difflib.get_close_matches(qt, list(meta.get('columns', [])), n=1, cutoff=0.9)
                        if close:
                            fuzzy += 1
            bm_norm = b / max_b
            final = 0.5 * exact_score + 0.3 * (col_hits + fuzzy) + 0.2 * bm_norm
            if meta.get('type') == 'table':
                final *= 1.3
            elif meta.get('type') == 'function':
                final *= 1.05
            if final > 0:
                scored.append((i, final))

        if not scored:
            return [], []

        scored.sort(key=lambda x: x[1], reverse=True)
        tables = [i for i, s in scored if self.meta[i]['type'] == 'table']
        funcs = [i for i, s in scored if self.meta[i]['type'] == 'function']
        others = [i for i, s in scored if self.meta[i]['type'] not in ('table','function')]

        selected = []
        for bucket in (tables, funcs, others):
            for idx in bucket:
                if idx not in selected:
                    selected.append(idx)
                if len(selected) >= top_k:
                    break
            if len(selected) >= top_k:
                break

        results = [self.docs[i] for i in selected]
        metas = [self.meta[i] for i in selected]
        return results, metas

    # ---------- Validation helpers ----------
    def extract_tables_from_sql(self, sql):
        back = re.findall(r'`([^`]+)`', sql)
        return set(self._normalize_identifier(b) for b in back)

    def extract_columns_from_sql(self, sql):
        cols = set()
        m = re.search(r'select\s+(.*?)\s+from', sql, re.I | re.S)
        if m:
            select_part = m.group(1)
            parts = select_part.split(',')
            for p in parts:
                p = p.strip()
                p = re.sub(r'\w+\(.*?\)', '', p)
                tokens = re.findall(r'`([^`]+)`|\b([A-Za-z_][A-Za-z0-9_]*)\b', p)
                for tk in tokens:
                    t = tk[0] or tk[1]
                    if t and not t.isdigit():
                        cols.add(t.lower())
        dotted = re.findall(r'([A-Za-z_][A-Za-z0-9_]*?)\.([A-Za-z_][A-Za-z0-9_]*)', sql)
        for a, b in dotted:
            cols.add(b.lower())
        return cols

    def extract_tables_from_schemas(self, metas):
        allowed = set()
        for m in metas:
            if m.get('type') == 'table' and m.get('fqtn'):
                allowed.add(self._normalize_identifier(m['fqtn']))
        return allowed

    def extract_schema_columns_map(self, metas):
        cmap = {}
        for m in metas:
            if m.get('type') == 'table' and m.get('fqtn'):
                key = self._normalize_identifier(m['fqtn'])
                cmap[key] = set(m.get('columns', set()))
        return cmap

    def validate_sql(self, sql, metas):
        allowed_tables = self.extract_tables_from_schemas(metas)
        allowed_cols_map = self.extract_schema_columns_map(metas)
        used_tables = self.extract_tables_from_sql(sql)
        used_cols = self.extract_columns_from_sql(sql)

        unknown_tables = [t for t in used_tables if t not in allowed_tables]
        if unknown_tables:
            return False, f"Unknown tables used: {unknown_tables}"

        allowed_cols_union = set().union(*allowed_cols_map.values()) if allowed_cols_map else set()
        if used_cols:
            bad_cols = [c for c in used_cols if c not in allowed_cols_union]
            if bad_cols:
                return False, f"Unknown columns used: {bad_cols}"

        return True, 'OK'

    def repair_sql(self, bad_sql, metas, user_msg, api_key, attempts=2):
        allowed = "\n".join([f"{m.get('fqtn','')}\ncolumns: {','.join(sorted(m.get('columns',[])))}" for m in metas if m.get('type')=='table'])
        system = (
            "You are a Senior BigQuery SQL engineer.\n"
            "Rewrite the provided SQL to use ONLY the tables and columns listed in the SCHEMA below.\n"
            "If it's impossible to satisfy the user request with the given schema, return exactly: -- ERROR: Insufficient schema\n"
        )
        for _ in range(attempts):
            prompt = f"{system}\nSCHEMA:\n{allowed}\n\nUser request:\n{user_msg}\n\nBAD_SQL:\n{bad_sql}\n\nReturn only the corrected SQL in a ```sql``` block."
            try:
                client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
                resp = client.chat(model=MODEL_NAME, messages=[{"role":"user","content":prompt}],
                                   options={"temperature":0.0, "top_p":0.8, "top_k":40})
                new_sql = resp.get('message',{}).get('content','')
                m = re.search(r'```sql\s*(.*?)```', new_sql, re.S | re.I)
                candidate = m.group(1).strip() if m else new_sql.strip()
                valid, reason = self.validate_sql(candidate, metas)
                if valid:
                    return True, candidate
            except Exception as e:
                print("Repair attempt error:", e)
        return False, 'Repair failed'

# =================== END RAGEngine ===================

# initialize
rag_engine = RAGEngine()
init_db()
rag_engine.load_schemas()

# =========================================================
# API
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    sessions = Session.query.order_by(desc(Session.created_at)).all()
    return jsonify([{'id': s.id, 'title': s.title} for s in sessions])

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    return jsonify(get_chat_history_formatted(session_id, limit=50))

@app.route('/api/reload', methods=['POST'])
def reload_schema():
    rag_engine.load_schemas()
    return jsonify({"status": "success", "message": "RAG Index Rebuilt!"})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    api_key = data.get('api_key') or DEFAULT_API_KEY
    user_msg = data.get('message')
    session_id = data.get('session_id')

    if not api_key or not session_id or not user_msg:
        return jsonify({"error": "Missing info"}), 400

    create_session_if_not_exists(session_id, user_msg)
    save_message(session_id, 'user', user_msg)

    # Step 1: expansion (conservative)
    expanded = rag_engine.expand_query(user_msg, api_key)

    # Step 2: retrieve
    results, metas = rag_engine.retrieve(user_msg, expanded, top_k=8)
    if not results:
        reply = "-- ERROR: Insufficient schema"
        save_message(session_id, 'assistant', reply)
        return jsonify({"response": reply})

    relevant_schemas_text = "\n\n".join(results)

    system_prompt = f"""
You are a Senior BigQuery SQL Architect.
STRICT RULES:
1) Use ONLY tables/columns listed in the SCHEMA below.
2) Do NOT invent any table or column.
3) Use fully-qualified table names with backticks `project.dataset.table`.
4) Return ONLY SQL inside a ```sql``` block. If impossible, return exactly: -- ERROR: Insufficient schema

SCHEMA:
{relevant_schemas_text}

User question:
{user_msg}
"""

    client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
    try:
        resp = client.chat(
            model=MODEL_NAME,
            messages=[{"role":"system","content":system_prompt}, {"role":"user","content":user_msg}],
            options={"temperature":0.0, "top_p":0.85, "top_k":20, "repeat_penalty":1.15, "num_ctx":8192}
        )
        raw = resp.get('message',{}).get('content','')
        m = re.search(r'```sql\s*(.*?)```', raw, re.S | re.I)
        sql_candidate = m.group(1).strip() if m else raw.strip()

        valid, reason = rag_engine.validate_sql(sql_candidate, metas)
        if not valid:
            ok, repaired = rag_engine.repair_sql(sql_candidate, metas, user_msg, api_key, attempts=2)
            if ok:
                final_sql = repaired
            else:
                final_sql = "-- ERROR: Insufficient schema"
        else:
            final_sql = sql_candidate

        save_message(session_id, 'assistant', final_sql)
        return jsonify({"response": final_sql})

    except Exception as e:
        print("Chat error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
