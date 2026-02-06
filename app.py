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
from sentence_transformers import SentenceTransformer
import numpy as np
# =========================================================
#  PHẦN 1: CONFIG & SETUP
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
HF_TOKEN = os.getenv("HF_TOKEN")

# =========================================================
#  PHẦN 2: DATABASE MODELS
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
#  PHẦN 3: ADVANCED RAG ENGINE (CORE LOGIC)
# =========================================================

# =========================================================
# HuggingFace Embedding API (NO LOCAL MODEL)
# =========================================================
def hf_embed(texts):
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-m3"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    response = requests.post(url, headers=headers, json={"inputs": texts})
    result = response.json()

    # ✅ Trích embedding từ dict nếu API trả về dict
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and 'embedding' in result[0]:
        embeddings_list = [item['embedding'] for item in result]
        return np.array(embeddings_list, dtype="float32")
    else:
        # fallback nếu trả thẳng list of list
        return np.array(result, dtype="float32")

# =========================================================
#  NEW HYBRID RAG ENGINE (BM25 + EMBEDDING + BOOST)
# =========================================================

class RAGEngine:
    def __init__(self):
        self.schema_docs = []
        self.doc_types = []          # table / function
        self.tokenized_corpus = []
        self.bm25 = None
        
        # 🔥 NEW: semantic search
        print("🔹 Using HuggingFace Embedding API (serverless)")
        self.embeddings = None
        
        self.is_ready = False

    # =====================================================
    # TOKENIZER (GIỮ NGUYÊN snake_case !!!)
    # =====================================================
    def tokenize(self, text):
        text = str(text)
        text = re.sub(r'[.\-\(\),`]', ' ', text)  # KHÔNG xóa _
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        tokens = text.lower().split()

        stopwords = {
            'string','int64','float','boolean','timestamp','date',
            'table','dataset','project','nullable','mode','type',
            'description','record','create','replace','function'
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    # =====================================================
    # LOAD SCHEMA + BUILD HYBRID INDEX
    # =====================================================
    def load_schemas(self):
        print("🚀 Building Hybrid RAG Index...")
        docs = []
        tokenized = []
        doc_types = []

        json_files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))

        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]

                for item in items:

                    # ================= TABLE =================
                    if 'table_name' in item:
                        dataset = item.get('table_schema')
                        project = "kynaforkids-server-production"
                        full_table = f"`{project}.{dataset}.{item['table_name']}`"

                        cols = []
                        raw_cols = item.get("columns", "[]")
                        parsed_cols = json.loads(raw_cols) if isinstance(raw_cols, str) else raw_cols
                        if isinstance(parsed_cols, list):
                            cols = parsed_cols

                        doc = f"""
[TABLE] {full_table}
Columns: {', '.join(cols)}

Business keywords:
teacher tutor student booking invoice complaint payment order lesson class
"""
                        keywords = f"{full_table} {' '.join(cols)}"

                        docs.append(doc)
                        tokenized.append(self.tokenize(keywords))
                        doc_types.append("table")

                    # ================= FUNCTION =================
                    elif 'routine_name' in item:
                        short_name = item['routine_name']
                        ddl = item.get('ddl', '')
                        definition = item.get('routine_definition', '')
                        arguments = item.get('arguments', '')

                        match = re.search(r'FUNCTION\s+`([^`]+)`', ddl, re.IGNORECASE)
                        full_name = match.group(1) if match else short_name

                        doc = f"""
[FUNCTION] `{full_name}`

Arguments: {arguments}

SQL Logic:
{definition}

Business purpose:
helper business logic mapping classification segmentation
teacher_type country nationality region geo filter mapping
"""
                        keywords = f"{full_name} {short_name} {definition}"

                        docs.append(doc)
                        tokenized.append(self.tokenize(keywords))
                        doc_types.append("function")

        # ===== BM25 =====
        self.bm25 = BM25Okapi(tokenized)

        # ===== EMBEDDINGS =====
        print("🔹 Creating embeddings via HF API...")
        self.embeddings = hf_embed(docs)

        self.schema_docs = docs
        self.doc_types = doc_types
        self.tokenized_corpus = tokenized
        self.is_ready = True

        print(f"✅ Indexed {len(docs)} schema objects")

    # =====================================================
    # QUERY EXPANSION (LLM)
    # =====================================================
    def query_expansion(self, query, api_key):
        try:
            client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
            prompt = f"""
Convert the user question into database search keywords.
Include:
- tables
- columns
- functions
- synonyms

User: {query}
Return keywords only.
"""
            res = client.chat(
                model=MODEL_NAME,
                messages=[{"role":"user","content":prompt}],
                options={"temperature":0}
            )
            return res['message']['content']
        except:
            return query

    # =====================================================
    # HYBRID RETRIEVAL (CORE)
    # =====================================================
    def retrieve(self, query, expanded_query, top_k=20):
        if not self.is_ready:
            return ""

        full_query = f"{query} {expanded_query}"

        # ---------- BM25 ----------
        bm25_scores = self.bm25.get_scores(self.tokenize(full_query))
        bm25_scores = np.array(bm25_scores)

        # ---------- VECTOR SEARCH ----------
        # ---------- VECTOR SEARCH (NO FAISS) ----------
        q_embed = hf_embed([full_query])[0]

        vector_scores = np.dot(self.embeddings, q_embed)

        # ---------- HYBRID SCORE ----------
        hybrid = 0.5 * bm25_scores + 0.5 * vector_scores

        # ---------- FUNCTION BOOST ----------
        if "func_" in full_query.lower():
            for i, t in enumerate(self.doc_types):
                if t == "function":
                    hybrid[i] *= 1.5

        # ---------- TOP K ----------
        top_idx = np.argsort(hybrid)[::-1][:top_k]
        results = [self.schema_docs[i] for i in top_idx if hybrid[i] > 0]

        return "\n----------------------\n".join(results)

# Khởi tạo RAG Engine
rag_engine = RAGEngine()
init_db()
rag_engine.load_schemas()

# =========================================================
#  PHẦN 4: API ROUTES
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

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    try:
        Message.query.delete()
        Session.query.delete()
        db.session.commit()
        return jsonify({"status": "success"})
    except:
        return jsonify({"status": "error"}), 500

@app.route("/api/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    try:
        Message.query.filter_by(session_id=session_id).delete()
        Session.query.filter_by(id=session_id).delete()
        db.session.commit()
        return jsonify({"status": "success"})
    except:
        return jsonify({"status": "error"}), 500

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

    if not api_key or not session_id:
        return jsonify({"error": "Missing info"}), 400

    create_session_if_not_exists(session_id, user_msg)
    save_message(session_id, "user", user_msg)

    try:
        # BƯỚC 1: QUERY EXPANSION (Làm giàu ngữ nghĩa)
        # Nếu câu hỏi quá ngắn, AI sẽ giúp đoán các bảng liên quan
        expanded_keywords = rag_engine.query_expansion(user_msg, api_key)
        
        # BƯỚC 2: BM25 RETRIEVAL (Tìm kiếm chính xác cao)
        # Chỉ lấy top 5 bảng liên quan nhất thay vì toàn bộ
        relevant_schemas = rag_engine.retrieve(user_msg, expanded_keywords, top_k=20)

        # BƯỚC 3: PROMPT ENGINEERING (Context-Aware Generation)
        system_prompt = f"""Role: Senior BigQuery SQL Architect.
Task: Generate an accurate Google BigQuery Standard SQL query strictly grounded in the provided schemas.

==================== CONTEXT ====================
{relevant_schemas}
=================================================

==================== HARD RULES (MUST FOLLOW) ====================

1. SOURCE OF TRUTH (NO HALLUCINATION)
- Use ONLY tables, columns, and routines that appear in CONTEXT.
- If a column/table is not present → DO NOT use it.
- Never assume generic columns (id, created_at, name, status, etc.).

2. EXACT TABLE NAMING
- Always use the FULLY QUALIFIED table name exactly as written in CONTEXT.
- Always wrap table names with backticks: `project.dataset.table`.

3. SCHEMA GROUNDING PROCESS (MANDATORY THINKING ORDER)
Before writing SQL:
Step 1 → Identify tables in CONTEXT that match the business question.
Step 2 → Identify exact columns that answer the question.
Step 3 → Verify joins only via columns that exist in schemas.
Step 4 → Only then generate SQL.

4. ROUTINE / FUNCTION LOGIC (CRITICAL)
If a FUNCTION/Routine appears in CONTEXT:
- Treat its SQL as the OFFICIAL business logic.
- Extract mappings from CASE WHEN / conditions.
- Convert business words → numeric/status codes using that logic.
- Routine can be used in SELECT or WHERE only.
- NEVER place routine inside FROM.

5. BIGQUERY BEST PRACTICES (MANDATORY)
- Use Google BigQuery Standard SQL.
- Prefer CTEs (WITH) for multi-step logic.
- Prefer JOIN instead of correlated subqueries.
- Avoid SELECT * → select only needed columns.
- Use SAFE_DIVIDE when dividing.
- Use explicit GROUP BY when aggregating.

6. OUTPUT FORMAT
- Return ONLY one SQL query inside a ```sql``` block.
- No markdown, no commentary outside the block.
- Short explanation AFTER the SQL is optional.

=================================================

User Question:
{user_msg}
"""

        messages_payload = [{"role": "system", "content": system_prompt}]
        history = get_chat_history_formatted(session_id, limit=6)
        
        for msg in history:
            if msg['content'] != user_msg: # Tránh duplicate
                messages_payload.append(msg)
        
        messages_payload.append({"role": "user", "content": user_msg})

        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        response = client.chat(
            model=MODEL_NAME,
            messages=messages_payload,
            stream=False,
            options={"temperature": 0.1} # Nhiệt độ thấp để code chính xác
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
