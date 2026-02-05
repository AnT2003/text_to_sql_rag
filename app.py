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
from rank_bm25 import BM25Okapi

# =========================================================
#  PHẦN 1: CONFIG & SETUP
# =========================================================

load_dotenv()
app = Flask(__name__)
CORS(app)

db_url = os.getenv("DATABASE_URL") or "sqlite:///local_chat.db"
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gpt-oss:120b" # Khuyến khích dùng model chuyên coder nếu có thể
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY")
SCHEMA_FOLDER = "./schemas"

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
    session_id = db.Column(db.String(50), db.ForeignKey('sessions.id', ondelete="CASCADE"), nullable=False)
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
#  PHẦN 3: ADVANCED RAG ENGINE (CORE INTELLIGENCE)
# =========================================================

class RAGEngine:
    def __init__(self):
        self.schema_docs = [] # Full text context
        self.bm25 = None      # BM25 Engine
        self.doc_map = {}     # Metadata map (Index -> Details)
        self.is_ready = False

    def tokenize(self, text):
        """ Tokenizer chuyên dụng cho Database: Giữ snake_case và tách word đơn. """
        text = str(text).lower()
        # Thay thế ký tự đặc biệt bằng khoảng trắng
        text = re.sub(r'[\.\-\(\)\,]', ' ', text)
        tokens = text.split()
        
        final_tokens = []
        for t in tokens:
            if len(t) < 2: continue
            final_tokens.append(t)
            # Nếu là snake_case (user_id), tách thêm thành ['user', 'id']
            if '_' in t:
                parts = t.split('_')
                final_tokens.extend([p for p in parts if len(p) > 1])
        return list(set(final_tokens))

    def load_schemas(self):
        print("🚀 Khởi tạo Advanced Schema Indexing...")
        new_docs, tokenized_corpus, self.doc_map = [], [], {}
        
        if not os.path.exists(SCHEMA_FOLDER): return

        json_files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        doc_content, keywords_source, is_view, short_name, obj_type = "", "", False, "", ""

                        # --- XỬ LÝ TABLE / VIEW ---
                        if 'table_name' in item:
                            short_name = item.get("table_name", "")
                            full_table_name = f"`{item.get('table_schema')}.{short_name}`"
                            raw_cols = item.get("columns", "[]")
                            cols = json.loads(raw_cols) if isinstance(raw_cols, str) else raw_cols
                            
                            doc_content = (
                                f"[TABLE] {full_table_name}\n"
                                f"Type: {item.get('table_type', 'UNKNOWN')}\n"
                                f"Columns: {', '.join(cols)}"
                            )
                            
                            is_view = 'view' in str(item.get('table_type', '')).lower()
                            obj_type = 'table'
                            # WEIGHTING: Nhân 5 lần tên bảng để ưu tiên search
                            keywords_source = (f"{short_name} " * 5) + " ".join(cols)

                        # --- XỬ LÝ ROUTINE / FUNCTION ---
                        elif 'routine_name' in item:
                            short_name = item.get('routine_name')
                            definition = item.get('routine_definition', '')
                            doc_content = f"[FUNCTION] {short_name}\nLogic:\n{definition}"
                            obj_type = 'function'
                            # Nhân đôi tên hàm
                            keywords_source = f"{short_name} {short_name} {definition}"

                        if doc_content:
                            idx = len(new_docs)
                            new_docs.append(doc_content)
                            tokenized_corpus.append(self.tokenize(keywords_source))
                            self.doc_map[idx] = {
                                'is_view': is_view, 
                                'short_name': short_name.lower(), 
                                'type': obj_type,
                                'table_type': item.get('table_type', 'BASE TABLE')
                            }
            except Exception as e: print(f"❌ Error: {e}")

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.schema_docs = new_docs
            self.is_ready = True
            print(f"✅ Index thành công {len(new_docs)} objects.")

    def query_expansion(self, user_query, api_key):
        """ Chuyển ngữ nghĩa người dùng sang thuật ngữ kỹ thuật Database. """
        try:
            client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
            prompt = f"""Task: Translate user intent to SQL technical keywords.
User Query: "{user_query}"
1. Identify tables/entities (e.g., 'khiếu nại' -> 'complain feedback report').
2. Identify conditions (e.g., 'hoàn thành' -> 'completed status').
3. OUTPUT ONLY technical keywords in English separated by spaces."""
            response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.0})
            keywords = re.sub(r'[^\w\s]', '', response['message']['content'])
            print(f"🔹 Expanded: {keywords}")
            return keywords
        except: return user_query

    def retrieve(self, query, expanded_query=None, top_k=25):
        """ Hybrid Retrieval: BM25 + Semantic Boosting + Table Priority. """
        if not self.is_ready: return ""
        
        search_query = f"{query} {expanded_query}"
        tokenized_query = self.tokenize(search_query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        candidates = []
        q_lower = query.lower()
        ex_lower = str(expanded_query).lower()
        
        for idx, score in enumerate(doc_scores):
            if score <= 0 and not any(kw in self.schema_docs[idx].lower() for kw in tokenized_query): continue
            
            meta = self.doc_map.get(idx, {})
            short_name = meta.get('short_name', '')
            boost = 0.0
            
            # RULE 1: EXACT NAME MATCH BOOSTING (Cực mạnh)
            if short_name and (short_name in q_lower or short_name in ex_lower):
                boost += 25.0
            
            # RULE 2: TYPE PRIORITY (Bắt buộc ưu tiên Table hơn View)
            if meta.get('type') == 'table':
                # Sửa đổi: Tăng khoảng cách boost giữa Table và View để buộc ưu tiên Table
                if not meta.get('is_view'): boost += 100.0 # Table gốc ưu tiên tuyệt đối
                else: boost -= 50.0 # View bị dìm điểm mạnh để xếp sau

            candidates.append((idx, score + boost))

        # Sắp xếp và lấy Top K
        candidates.sort(key=lambda x: x[1], reverse=True)
        results = [self.schema_docs[i] for i, s in candidates[:top_k]]
        
        if not results: return "\n".join(self.schema_docs[:3])
        return "\n--------------------\n".join(results)

# Khởi tạo Engine
rag_engine = RAGEngine()

# =========================================================
#  PHẦN 4: API & CHAT LOGIC
# =========================================================

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    api_key, user_msg, session_id = data.get('api_key') or DEFAULT_API_KEY, data.get('message'), data.get('session_id')
    
    if not api_key or not session_id: return jsonify({"error": "Missing info"}), 400
    create_session_if_not_exists(session_id, user_msg)
    save_message(session_id, "user", user_msg)

    try:
        # BƯỚC 1: Mở rộng từ khóa (Dịch Việt -> Anh kỹ thuật)
        expanded = rag_engine.query_expansion(user_msg, api_key)
        
        # BƯỚC 2: Tìm kiếm Schema (Ưu tiên bảng, lọc nhiễu)
        context = rag_engine.retrieve(user_msg, expanded, top_k=25)

        # BƯỚC 3: Prompt Engineering (Chain-of-Thought)
        system_prompt = f"""Role: Senior BigQuery SQL Architect.
Goal: Generate optimized Google BigQuery SQL.

[CONTEXT - DATABASE SCHEMAS]:
{context}

[STRICT RULES - NO EXCEPTIONS]:
1. SOURCE OF TRUTH: Use ONLY tables and columns listed in [CONTEXT]. 
2. TABLE PRIORITY: You MUST prioritize selecting from BASE TABLE over VIEW if the required data is available in both. Using a VIEW is only permitted if the data cannot be found in any BASE TABLE.
3. LOGIC GROUNDING: For statuses (e.g., 'Completed', 'Vietnam'), you MUST check [FUNCTION] / Routine definitions in Context to find exact ID values.
4. TABLE NAMING: Use backticks for full names: `project.dataset.table`.

[RESPONSE FORMAT]:
**1. Logical Analysis:**
- Tables to use: (Must exist in Context)
- Business Logic: (Map text to IDs using Routines)
**2. SQL Query:**
```sql
-- Your Query Here
```
"""
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        response = client.chat(
            model=MODEL_NAME, 
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            options={"temperature": 0.0} # Tuyệt đối không sáng tạo
        )
        
        reply = response['message']['content']
        save_message(session_id, "assistant", reply)
        return jsonify({"response": reply})

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": str(e)}), 500

# Các route khác (giữ nguyên logic gốc của bạn)
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    sessions = Session.query.order_by(desc(Session.created_at)).all()
    return jsonify([{'id': s.id, 'title': s.title} for s in sessions])

@app.route('/api/reload', methods=['POST'])
def reload_schema():
    rag_engine.load_schemas()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    init_db()
    rag_engine.load_schemas()
    app.run(debug=True, port=5000)
