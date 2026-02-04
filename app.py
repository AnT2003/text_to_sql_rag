# ================= FULL PRECISION-FIRST RAG SQL CHAT APP =================
# Mục tiêu: chống hallucination tối đa nhưng KHÔNG làm thay đổi API đã deploy

import os
import json
import glob
import datetime
import re
from collections import defaultdict

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from rank_bm25 import BM25Okapi

# =========================================================
#  PHẦN 1: CONFIG & SETUP (GIỮ NGUYÊN)
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

OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gemini-3-flash-preview:latest"
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY")
SCHEMA_FOLDER = "./schemas"
PROJECT_ID = "kynaforkids-server-production"

# =========================================================
#  PHẦN 2: DATABASE MODELS (GIỮ NGUYÊN)
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
    except:
        db.session.rollback()


def create_session_if_not_exists(session_id, first_msg):
    if not Session.query.get(session_id):
        title = (first_msg[:50] + '...') if len(first_msg) > 50 else first_msg
        db.session.add(Session(id=session_id, title=title))
        db.session.commit()


def get_chat_history_formatted(session_id, limit=10):
    msgs = Message.query.filter_by(session_id=session_id) \
        .order_by(desc(Message.created_at)).limit(limit).all()
    return [{"role": m.role, "content": m.content} for m in msgs[::-1]]

# =========================================================
#  PHẦN 3: ULTRA PRECISION RAG ENGINE (CHỈ PHẦN NÀY ĐƯỢC NÂNG CẤP)
# =========================================================

class RAGEngine:
    def __init__(self):
        self.docs = []
        self.tokens = []
        self.meta = {}
        self.bm25 = None
        self.ready = False

    # Tokenizer SQL aware (snake_case + camelCase + loại noise)
    def tokenize(self, text: str):
        text = re.sub(r'[`._\-(),]', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        tokens = text.lower().split()
        stop = {'string','int64','float','float64','boolean','timestamp','date','nullable','create','replace','view','table','dataset','project'}
        return [t for t in tokens if t not in stop and len(t) > 1]

    # Load schema STRICT từ DDL (không suy đoán dataset nữa)
    def load_schemas(self):
        self.docs, self.tokens, self.meta = [], [], {}
        files = glob.glob(os.path.join(SCHEMA_FOLDER, '*.json'))

        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]

                for item in items:

                    # -------- TABLE / VIEW --------
                    if 'table_name' in item:
                        ddl = item.get('ddl', '')
                        match = re.search(r'`([^`]+)`', ddl)
                        if not match:
                            continue
                        fqtn = f"`{match.group(1)}`"

                        raw_cols = item.get('columns', [])
                        if isinstance(raw_cols, str):
                            try: raw_cols = json.loads(raw_cols)
                            except: raw_cols = []
                        cols = raw_cols if isinstance(raw_cols, list) else []

                        doc = "[TABLE] " + fqtn + "\nColumns:\n" + "\n".join(f"- {c}" for c in cols)
                        keywords = " ".join([fqtn] + cols)

                        idx = len(self.docs)
                        self.docs.append(doc)
                        self.tokens.append(self.tokenize(keywords))
                        self.meta[idx] = {
                            "type": "table",
                            "fqtn": fqtn,
                            "columns": set(c.lower() for c in cols)
                        }

                    # -------- FUNCTION --------
                    elif 'routine_name' in item:
                        ddl = item.get('ddl', '')
                        match = re.search(r'`([^`]+)`', ddl)
                        fname = f"`{match.group(1)}`" if match else item['routine_name']
                        definition = item.get('routine_definition', '')

                        doc = f"[FUNCTION] {fname}\nLogic:\n{definition}"
                        keywords = fname + " " + definition

                        idx = len(self.docs)
                        self.docs.append(doc)
                        self.tokens.append(self.tokenize(keywords))
                        self.meta[idx] = {"type": "function"}

        if self.tokens:
            self.bm25 = BM25Okapi(self.tokens)
            self.ready = True
            print(f"✅ RAG indexed {len(self.docs)} docs")

    # Expansion chỉ semantic (không sinh tên bảng/cột)
    def expand_query(self, query, api_key):
        try:
            client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
            prompt = f"""
Return generic analytics keywords related to:
{query}
Rules:
- Do NOT generate table names
- Do NOT generate column names
Comma separated only.
"""
            r = client.chat(model=MODEL_NAME, messages=[{"role":"user","content":prompt}], options={"temperature":0})
            return r['message']['content']
        except:
            return query

    # Hybrid retrieval + heavy column boosting
    def retrieve(self, query, expanded, top_k=10):
        if not self.ready:
            return ""

        q_tokens = self.tokenize(query + " " + (expanded or ""))
        scores = self.bm25.get_scores(q_tokens)

        ranked = []
        for i, base in enumerate(scores):
            if base <= 0: continue
            boost = 0
            if self.meta[i]['type'] == 'table':
                boost += len(set(q_tokens) & self.meta[i]['columns']) * 2.5
                if set(q_tokens) & set(self.tokenize(self.meta[i]['fqtn'])):
                    boost += 3
            ranked.append((i, base + boost))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return "\n--------------------\n".join(self.docs[i] for i,_ in ranked[:top_k])

# =========================================================
#  INIT
# =========================================================

rag_engine = RAGEngine()
init_db()
rag_engine.load_schemas()

# =========================================================
#  API ROUTES (GIỮ NGUYÊN, CHỈ UPDATE PROMPT)
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    api_key = data.get('api_key') or DEFAULT_API_KEY
    user_msg = data.get('message')
    session_id = data.get('session_id')

    if not api_key or not session_id:
        return jsonify({'error': 'Missing info'}), 400

    create_session_if_not_exists(session_id, user_msg)
    save_message(session_id, 'user', user_msg)

    expanded = rag_engine.expand_query(user_msg, api_key)
    schemas = rag_engine.retrieve(user_msg, expanded)

    if not schemas:
        reply = "Không tìm thấy bảng phù hợp trong schema."
        save_message(session_id, 'assistant', reply)
        return jsonify({'response': reply})

    system_prompt = f"""
You are a Senior BigQuery SQL Architect.

MANDATORY RULES:
1. Use ONLY tables, columns and routines in [SCHEMA].
2. If missing → return INSUFFICIENT_SCHEMA.
3. NEVER invent joins or columns.
4. Reuse function logic EXACTLY if provided.
5. Use FULLY QUALIFIED names with backticks.
6. Output ONLY SQL inside ```sql```.

[SCHEMA]
{schemas}

User Question:
{user_msg}
"""

    client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
    resp = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
        options={"temperature": 0.05}
    )

    reply = resp['message']['content']
    save_message(session_id, 'assistant', reply)
    return jsonify({'response': reply})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
