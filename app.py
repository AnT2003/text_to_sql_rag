import os
import json
import glob
import datetime
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from rank_bm25 import BM25Okapi  # <--- THƯ VIỆN RAG MẠNH MẼ
import requests
import numpy as np

# =========================================================
# PHẦN 1: CONFIG & SETUP
# =========================================================

load_dotenv()
app = Flask(__name__)
CORS(app)

db_url = os.getenv("DATABASE_URL")
if not db_url:
    db_url = "sqlite:///local_chat.db"
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gpt-oss:120b"
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY")
SCHEMA_FOLDER = "./schemas"

# =========================================================
# PHẦN 1B: OpenRouter Embedding (thay HF cũ)
# =========================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 

def openrouter_embedding(texts, model="sentence-transformers/all-minilm-l12-v2"):
    """
    Trả về numpy array embeddings từ OpenRouter API
    """
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",  # optional
        "X-Title": "Data Analysis Project"        # optional
    }
    payload = {
        "model": model,
        "input": texts
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code != 200:
        raise ValueError(f"OpenRouter Error [{res.status_code}]: {res.text}")
    response_data = res.json()
    embeddings = np.array([item["embedding"] for item in response_data["data"]], dtype="float32")

    # ⭐ normalize để dùng cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    return embeddings

# =========================================================
# PHẦN 2: DATABASE MODELS
# =========================================================

class Session(db.Model):
    __tablename__ = 'sessions'
    id = db.Column(db.String(50), primary_key=True)
    title = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), db.ForeignKey('sessions.id',ondelete="CASCADE"), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

def init_db():
    with app.app_context():
        db.create_all()

def save_message(session_id, role, content):
    try:
        new_msg = Message(session_id=session_id, role=role, content=content)
        db.session.add(new_msg)
        db.session.commit()
    except:
        db.session.rollback()

def create_session_if_not_exists(session_id, first_msg):
    try:
        if not Session.query.get(session_id):
            title = (first_msg[:50] + '...') if len(first_msg) > 50 else first_msg
            db.session.add(Session(id=session_id, title=title))
            db.session.commit()
    except:
        db.session.rollback()

def get_chat_history_formatted(session_id, limit=10):
    try:
        msgs = Message.query.filter_by(session_id=session_id).order_by(desc(Message.created_at)).limit(limit).all()
        return [{"role": m.role, "content": m.content} for m in msgs[::-1]]
    except:
        return []

# =========================================================
# PHẦN 3: ADVANCED RAG ENGINE (TECH UPGRADE: DIRECT SCHEMA LINKING)
# =========================================================

def hf_embed(texts):
    return openrouter_embedding(texts)

class RAGEngine:
    def __init__(self):
        self.schema_docs = []
        self.schema_metadata = [] # Lưu thông tin định danh: {name, type}
        self.doc_types = []       # table / function
        self.tokenized_corpus = []
        self.bm25 = None
        self.embeddings = None
        self.is_ready = False

    def tokenize(self, text):
        text = str(text)
        # GIỮ NGUYÊN snake_case cho tên bảng, chỉ tách ký tự đặc biệt khác
        text = re.sub(r'[.\-\(\),`]', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        tokens = text.lower().split()

        stopwords = {
            'string','int64','float','boolean','timestamp','date',
            'table','dataset','project','nullable','mode','type',
            'description','record','create','replace','function'
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def load_schemas(self):
        print("🚀 Building Advanced Hybrid RAG Index...")
        docs = []
        tokenized = []
        metadata = []
        doc_types = []

        json_files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))

        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]

                for item in items:
                    if 'table_name' in item:
                        name = item['table_name']
                        dataset = item.get('table_schema', 'kynaforkids')
                        project = "kynaforkids-server-production"
                        full_table = f"`{project}.{dataset}.{name}`"

                        raw_cols = item.get("columns", "[]")
                        cols = json.loads(raw_cols) if isinstance(raw_cols, str) else raw_cols
                        
                        doc = f"[TABLE] {full_table}\nCOLUMNS: {', '.join(cols)}"
                        docs.append(doc)
                        tokenized.append(self.tokenize(f"{name} {' '.join(cols)}"))
                        metadata.append({"name": name, "type": "table", "full": full_table})
                        doc_types.append("table")

                    elif 'routine_name' in item:
                        name = item['routine_name']
                        definition = item.get('routine_definition', '')
                        args = item.get('arguments', '')
                        full_name = f"`kynaforkids-server-production.kynaforkids.{name}`"
                        
                        doc = f"[FUNCTION] {full_name}\nARGS: {args}\nLOGIC: {definition}"
                        docs.append(doc)
                        tokenized.append(self.tokenize(f"{name} {definition}"))
                        metadata.append({"name": name, "type": "function", "full": full_name})
                        doc_types.append("function")

        self.bm25 = BM25Okapi(tokenized)
        self.embeddings = hf_embed(docs)
        self.schema_docs = docs
        self.schema_metadata = metadata
        self.doc_types = doc_types
        self.is_ready = True
        print(f"✅ Indexed {len(docs)} schema objects")

    def query_expansion(self, query, api_key):
        try:
            client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
            prompt = f"Identify potential table names, columns, and functions for: {query}. Return keywords only."
            res = client.chat(model=MODEL_NAME, messages=[{"role":"user","content":prompt}], options={"temperature":0})
            return res['message']['content']
        except:
            return query

    def retrieve(self, query, expanded_query, top_k=15):
        if not self.is_ready: return ""
        full_query = f"{query} {expanded_query}".lower()

        # 1. BM25 & Vector Search
        bm25_scores = np.array(self.bm25.get_scores(self.tokenize(full_query)))
        q_embed = hf_embed([full_query])[0]
        q_embed = q_embed / np.clip(np.linalg.norm(q_embed), 1e-12, None)
        vector_scores = np.dot(self.embeddings, q_embed)

        # 2. Hybrid Score (Normalized)
        if np.max(bm25_scores) > 0: bm25_scores /= np.max(bm25_scores)
        hybrid_scores = 0.5 * bm25_scores + 0.5 * vector_scores

        # 3. DIRECT SCHEMA LINKING (The "LLM Notebook" technique)
        # Nếu câu hỏi chứa chính xác tên bảng hoặc tên hàm, ta Boost cực mạnh
        for i, meta in enumerate(self.schema_metadata):
            if meta['name'].lower() in full_query:
                hybrid_scores[i] += 2.0 # Boost mạnh để bảng đúng luôn đứng đầu
            
            # Boost thêm cho function nếu query có keywords nghiệp vụ
            if meta['type'] == 'function' and any(kw in full_query for kw in ['country', 'status', 'rating']):
                hybrid_scores[i] += 0.5

        top_idx = np.argsort(hybrid_scores)[::-1][:top_k]
        results = [self.schema_docs[i] for i in top_idx if hybrid_scores[i] > 0.1]
        return "\n----------------------\n".join(results)

# Khởi tạo
rag_engine = RAGEngine()
init_db()
rag_engine.load_schemas()

# =========================================================
# PHẦN 4: API ROUTES
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
        return jsonify({"error": "Missing info"}), 400

    create_session_if_not_exists(session_id, user_msg)
    save_message(session_id, "user", user_msg)

    try:
        # Step 1: Query Expansion
        expanded = rag_engine.query_expansion(user_msg, api_key)
        
        # Step 2: Advanced Retrieval (với Direct Linking Boost)
        relevant_schemas = rag_engine.retrieve(user_msg, expanded)

        # Step 3: Prompt Engineering
        system_prompt = f"""Role: Senior BigQuery SQL Architect.
Task: Generate an accurate Google BigQuery Standard SQL query based on the provided Context.

==================== CONTEXT (SOURCE OF TRUTH) ====================
{relevant_schemas}
===================================================================

==================== STRICT RULES ====================
1. MANDATORY TABLE LINKING:
   - Identify if the user mentions a specific table (e.g., 'booking_session'). 
   - If 'booking_session' is requested, you MUST use it. Do NOT use 'Test_student_teacher' for booking logic.
2. FUNCTION LOGIC:
   - Use provided functions for country, rating, or status logic.
   - Example: `dataset.func_get_country(id) = 'Việt Nam'`.
3. SYNTAX:
   - Use backticks: `project.dataset.table`.
   - Return ONLY the SQL block and a short logic explanation.
4. NO HALLUCINATION: If a column is not in Context, do not use it.
======================================================

User Question: {user_msg}
"""

        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            options={"temperature": 0}
        )
        
        reply = response['message']['content']
        save_message(session_id, "assistant", reply)
        return jsonify({"response": reply})

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    rag_engine.load_schemas()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
