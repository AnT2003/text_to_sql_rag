import os
import json
import glob
import datetime
import re
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from rank_bm25 import BM25Okapi

# =========================================================
# CONFIG & SETUP
# =========================================================
load_dotenv()
app = Flask(__name__)
CORS(app)

db_url = os.getenv("DATABASE_URL", "sqlite:///local_chat.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

OLLAMA_HOST = "https://ollama.com" # Hoặc endpoint local của bạn
MODEL_NAME = "gpt-oss:120b"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SCHEMA_FOLDER = "./schemas"

# =========================================================
# UTILS & EMBEDDING
# =========================================================
def openrouter_embedding(texts, model="sentence-transformers/all-minilm-l12-v2"):
    if not OPENROUTER_API_KEY:
        # Nếu không có key, trả về vector 0 để tránh crash, nhưng khuyến khích config key
        return np.zeros((len(texts), 768), dtype="float32")
    
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "input": texts}
    
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code != 200:
            return np.zeros((len(texts), 768), dtype="float32")
        
        data = res.json()
        embeddings = np.array([item["embedding"] for item in data["data"]], dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-12, None)
    except:
        return np.zeros((len(texts), 768), dtype="float32")

# =========================================================
# DATABASE MODELS
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

# =========================================================
# ADVANCED RAG ENGINE (TECH UPGRADE)
# =========================================================
class AdvancedRAGEngine:
    def __init__(self):
        self.schema_metadata = [] # Lưu object chi tiết
        self.schema_docs = []     # Lưu text để search
        self.bm25 = None
        self.embeddings = None
        self.is_ready = False

    def tokenize(self, text):
        # Giữ nguyên snake_case cực kỳ quan trọng cho tên bảng SQL
        text = str(text).lower()
        # Loại bỏ ký tự đặc biệt nhưng giữ dấu gạch dưới
        text = re.sub(r'[^a-z0-0_\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]

    def load_schemas(self):
        print("🚀 Building Advanced Schema Index...")
        docs, metadata, tokenized_corpus = [], [], []
        json_files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))

        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    items = json.load(f)
                    if not isinstance(items, list): items = [items]

                    for item in items:
                        if 'table_name' in item:
                            name = item['table_name']
                            dataset = item.get('table_schema', 'kynaforkids')
                            full_table = f"`kynaforkids-server-production.{dataset}.{name}`"
                            
                            raw_cols = item.get("columns", "[]")
                            cols = json.loads(raw_cols) if isinstance(raw_cols, str) else raw_cols
                            
                            # Enrichment: Thêm nhiều từ khóa nghiệp vụ vào doc search
                            doc_content = f"TABLE_ENTITY: {name} FULL_PATH: {full_table} COLUMNS: {' '.join(cols)} KEYWORDS: {name.replace('_', ' ')}"
                            docs.append(doc_content)
                            tokenized_corpus.append(self.tokenize(doc_content))
                            metadata.append({"type": "table", "name": name, "full": full_table, "cols": cols})

                        elif 'routine_name' in item:
                            name = item['routine_name']
                            definition = item.get('routine_definition', '')
                            full_func = f"`kynaforkids-server-production.kynaforkids.{name}`"
                            
                            doc_content = f"FUNCTION_ENTITY: {name} LOGIC: {definition} KEYWORDS: {name.replace('_', ' ')}"
                            docs.append(doc_content)
                            tokenized_corpus.append(self.tokenize(doc_content))
                            metadata.append({"type": "function", "name": name, "full": full_func, "definition": definition})
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        if docs:
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.embeddings = openrouter_embedding(docs)
            self.schema_docs = docs
            self.schema_metadata = metadata
            self.is_ready = True
        print(f"✅ Indexed {len(docs)} objects.")

    def retrieve(self, query, top_k=10):
        if not self.is_ready: return ""

        # 1. Hybrid Search (BM25 + Vector)
        tokens = self.tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(tokens))
        
        q_embed = openrouter_embedding([query])[0]
        vector_scores = np.dot(self.embeddings, q_embed)
        
        # Normalize scores về 0-1
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)
        
        # 2. Advanced Scoring Logic
        # Kết hợp trọng số: BM25 (từ khóa chính xác) + Vector (ý nghĩa ngữ nghĩa)
        combined_scores = 0.5 * bm25_scores + 0.5 * vector_scores

        # 3. Entity-Based Re-ranking (Cực kỳ quan trọng để sửa lỗi chọn sai bảng)
        # Nếu người dùng nhắc tên bảng gần đúng hoặc chính xác, ta ép bảng đó lên đầu
        query_clean = query.lower()
        for i, meta in enumerate(self.schema_metadata):
            name_lower = meta['name'].lower()
            # Kiểm tra match chính xác hoặc match một phần quan trọng (ví dụ 'booking' trong 'booking_session')
            if name_lower in query_clean or any(part in query_clean for part in name_lower.split('_') if len(part) > 3):
                combined_scores[i] += 1.5 # Boost cực mạnh cho bảng/hàm trùng tên
            
            # Boost thêm nếu là FUNCTION và người dùng hỏi về logic mapping/country/status
            if meta['type'] == 'function' and any(kw in query_clean for kw in ['country', 'status', 'rating', 'loại', 'phân loại']):
                combined_scores[i] += 0.5

        # 4. Lọc và định dạng kết quả
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        formatted_results = []
        for idx in top_indices:
            if combined_scores[idx] < 0.1: continue # Ngưỡng tối thiểu
            m = self.schema_metadata[idx]
            if m['type'] == 'table':
                formatted_results.append(f"### TABLE: {m['full']}\n- Columns: {', '.join(m['cols'])}")
            else:
                # Chỉ lấy phần logic quan trọng nhất của hàm để tiết kiệm context
                definition = m['definition']
                if len(definition) > 1000: definition = definition[:1000] + "..."
                formatted_results.append(f"### FUNCTION: {m['full']}\n- Logic:\n{definition}")

        return "\n\n".join(formatted_results)

rag_engine = AdvancedRAGEngine()

# =========================================================
# API ROUTES
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
    try:
        msgs = Message.query.filter_by(session_id=session_id).order_by(Message.created_at).all()
        return jsonify([{"role": m.role, "content": m.content} for m in msgs])
    except:
        return jsonify([])

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message')
    session_id = data.get('session_id')
    api_key = data.get('api_key') or os.getenv("OLLAMA_API_KEY")

    if not session_id: return jsonify({"error": "Missing session_id"}), 400

    # Lưu message user
    new_msg = Message(session_id=session_id, role="user", content=user_msg)
    db.session.add(new_msg)
    db.session.commit()

    # 1. Retrieval với kỹ thuật nâng cao
    context = rag_engine.retrieve(user_msg)

    # 2. Refined Prompt Engineering (Chain of Thought for SQL)
    system_prompt = f"""You are an Expert BigQuery SQL Architect.
Your task is to generate a Standard SQL query based ONLY on the provided schemas.

### SCHEMA CONTEXT:
{context}

### STRICT INSTRUCTIONS:
1. SCHEMA LINKING (MANDATORY):
   - First, identify which tables and functions from the CONTEXT are actually relevant to the question.
   - If the user explicitly mentions a table name like 'booking_session', you MUST prioritize it over other tables like 'Test_student_teacher'.
   - NEVER use a table or column that is not in the context.

2. BUSINESS LOGIC & FUNCTIONS:
   - Use the provided FUNCTIONS for any logic related to country, rating, status, or types.
   - Example: If a function `func_get_booking_status` exists, use it in the WHERE or SELECT clause.

3. BIGQUERY STANDARDS:
   - Always use fully qualified table names with backticks: `project.dataset.table`.
   - Use CTEs (WITH clause) for complex logic.
   - Use SAFE_DIVIDE for divisions.

### OUTPUT FORMAT:
- Start with a brief "Reasoning" section explaining which tables/functions you chose and why.
- Then provide the SQL query in a ```sql block.
"""

    try:
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            options={"temperature": 0.1}
        )
        
        reply = response['message']['content']
        
        # Lưu message assistant
        assistant_msg = Message(session_id=session_id, role="assistant", content=reply)
        db.session.add(assistant_msg)
        db.session.commit()
        
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    rag_engine.load_schemas()
    app.run(host="0.0.0.0", port=10000)
