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
# Lưu ý: Nên dùng model chuyên code như qwen2.5-coder hoặc deepseek-coder nếu có thể
MODEL_NAME = "gpt-oss:120b" 
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
#  PHẦN 3: ADVANCED RAG ENGINE (CORE LOGIC - OPTIMIZED)
# =========================================================

class RAGEngine:
    def __init__(self):
        self.schema_docs = [] # Lưu nội dung text full
        self.bm25 = None      # Object tìm kiếm BM25
        self.doc_map = {}     # Map index -> metadata (để scoring lại)
        self.is_ready = False

    def tokenize(self, text):
        """
        Tokenization nâng cao:
        Giữ nguyên cụm từ snake_case VÀ tách rời chúng.
        Ví dụ: "order_details" -> ["order_details", "order", "details"]
        Để search chính xác cũng trúng mà search từ đơn cũng trúng.
        """
        text = str(text).lower()
        # Tách thô bằng ký tự đặc biệt
        raw_tokens = re.split(r'[\s\.\-\(\)\,]+', text)
        
        final_tokens = []
        stopwords = {
            'string', 'int64', 'float', 'boolean', 'timestamp', 'date', 
            'table', 'dataset', 'project', 'nullable', 'mode', 'type', 
            'description', 'record', 'create', 'replace', 'partition', 'by',
            'select', 'from', 'where'
        }

        for t in raw_tokens:
            if not t or t in stopwords: continue
            
            final_tokens.append(t) # Giữ nguyên (vd: user_id)
            
            # Nếu có snake_case, tách thêm (vd: user, id)
            if '_' in t:
                sub_tokens = t.split('_')
                final_tokens.extend([st for st in sub_tokens if len(st) > 1])
        
        return final_tokens

    def load_schemas(self):
        print("🚀 Đang khởi tạo Advanced RAG Indexing (Optimized)...")
        new_docs = []
        tokenized_corpus = []
        self.doc_map = {} # Reset map
        
        if not os.path.exists(SCHEMA_FOLDER):
            print("⚠️ Không tìm thấy folder schemas")
            return

        json_files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else [data]

                    for item in items:
                        doc_content = ""
                        keywords_source = ""
                        is_view = False
                        short_name = ""
                        obj_type = ""

                        # ===============================
                        # CASE 1: TABLES / VIEWS
                        # ===============================
                        if 'table_name' in item:
                            short_name = item.get("table_name", "")
                            dataset_id = item.get('table_schema')
                            project_id = 'kynaforkids-server-production'
                            
                            # Prefix đầy đủ
                            full_prefix = f"{project_id}.{dataset_id}" if project_id else dataset_id
                            full_table_name = f"`{full_prefix}.{short_name}`"

                            # Parse columns
                            cols_desc = []
                            col_tokens = []
                            raw_cols = item.get("columns", "[]")
                            try:
                                parsed_cols = json.loads(raw_cols) if isinstance(raw_cols, str) else raw_cols
                                if isinstance(parsed_cols, list):
                                    for col in parsed_cols:
                                        cols_desc.append(f"- {col}")
                                        col_tokens.append(col)
                            except: pass

                            # Doc content display cho LLM
                            doc_content = (
                                f"[TABLE] {full_table_name}\n"
                                f"Type: {item.get('table_type', 'UNKNOWN')}\n"
                                f"Columns:\n" + "\n".join(cols_desc)
                            )
                            
                            # Metadata
                            table_type_str = str(item.get('table_type', '')).lower()
                            is_view = 'view' in table_type_str
                            obj_type = 'table'

                            # WEIGHTING: Nhân 3 tên bảng để BM25 ưu tiên nó hơn tên cột
                            keywords_source = (f"{short_name} " * 3) + f"{full_table_name} " + " ".join(col_tokens)

                        # ===============================
                        # CASE 2: FUNCTIONS / ROUTINES
                        # ===============================
                        elif 'routine_name' in item:
                            short_name = item.get('routine_name')
                            match = re.search(r'FUNCTION\s+`([^`]+)`', item.get('ddl', ''), re.IGNORECASE)
                            full_name = f"`{match.group(1)}`" if match else f"`{short_name}`"
                            definition = item.get('routine_definition', '')
                            
                            doc_content = f"[FUNCTION] {full_name}\nLogic:\n{definition}"
                            
                            # Metadata
                            is_view = False
                            obj_type = 'function'
                            
                            # Weighting: Nhân đôi tên function
                            keywords_source = f"{short_name} {short_name} {definition}"

                        # ===============================
                        # ADD TO CORPUS
                        # ===============================
                        if doc_content:
                            current_idx = len(new_docs)
                            new_docs.append(doc_content)
                            tokenized_corpus.append(self.tokenize(keywords_source))
                            
                            # Lưu metadata để Re-ranking sau này
                            self.doc_map[current_idx] = {
                                'is_view': is_view,
                                'short_name': str(short_name).lower(),
                                'type': obj_type
                            }

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # BUILD INDEX
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.schema_docs = new_docs
            self.is_ready = True
            print(f"✅ Đã index {len(new_docs)} schemas.")
        else:
            print("⚠️ Không có dữ liệu để index.")

    def query_expansion(self, user_query, api_key):
        """
        Dùng AI để tìm từ khóa đồng nghĩa, tránh hallucination ngay từ bước này.
        """
        try:
            client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
            prompt = f"""Task: Extract database keywords from user query.
User Query: "{user_query}"
Instructions:
1. Identify potential table names (e.g., if user says 'users', output 'users').
2. Identify potential column names (e.g., 'revenue', 'id').
3. Return ONLY a list of English keywords separated by spaces.
4. Do NOT invent specific table names if not sure."""
            
            response = client.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0}
            )
            keywords = response['message']['content']
            print(f"🔹 Expanded Query: {keywords}")
            return keywords
        except:
            return user_query

    def retrieve(self, query, expanded_query=None, top_k=10):
        """
        Hybrid Retrieval (BM25 + Rules):
        1. Lấy Top 50 bằng BM25.
        2. Chấm điểm lại (Re-ranking) dựa trên rules:
           - Trùng tên bảng: +15 điểm.
           - Là Table: +5 điểm.
           - Là View: -2 điểm.
        3. Lấy Top K.
        """
        if not self.is_ready: return ""
        
        search_query = f"{query} {expanded_query}" if expanded_query else query
        tokenized_query = self.tokenize(search_query)
        
        # 1. Raw Scores từ BM25
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. Lấy danh sách sơ bộ (Candidate generation)
        top_n_candidates = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:50]
        
        final_scored_candidates = []
        query_lower = query.lower()
        
        for idx in top_n_candidates:
            original_score = doc_scores[idx]
            if original_score <= 0: continue # Bỏ rác
            
            metadata = self.doc_map.get(idx, {})
            short_name = metadata.get('short_name', '')
            is_view = metadata.get('is_view', False)
            obj_type = metadata.get('type')
            
            # 3. Custom Scoring (Re-ranking)
            boost = 0.0
            
            # Rule A: Name Match (Quan trọng nhất)
            # Nếu tên bảng (vd: 'users') xuất hiện chính xác trong query -> Boost mạnh
            if short_name and re.search(r'\b' + re.escape(short_name) + r'\b', query_lower):
                boost += 15.0 
            
            # Rule B: Object Type Priority
            if obj_type == 'table' and not is_view:
                boost += 5.0  # Ưu tiên Table gốc
            elif is_view:
                boost -= 2.0  # Giảm điểm View để Table gốc nổi lên
                
            final_score = original_score + boost
            final_scored_candidates.append((idx, final_score))
            
        # 4. Sort theo Final Score và lấy Top K
        final_scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = [self.schema_docs[idx] for idx, score in final_scored_candidates[:top_k]]
        
        if not results:
            print("⚠️ Không tìm thấy schema khớp, lấy default top 2.")
            return "\n".join(self.schema_docs[:2])
            
        return "\n--------------------\n".join(results)

# Khởi tạo RAG Engine global
rag_engine = RAGEngine()

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
        # BƯỚC 1: QUERY EXPANSION
        expanded_keywords = rag_engine.query_expansion(user_msg, api_key)
        
        # BƯỚC 2: HYBRID RETRIEVAL (Đã tối ưu logic ưu tiên Table)
        # Tăng top_k lên 15 để đảm bảo LLM có đủ context lựa chọn
        relevant_schemas = rag_engine.retrieve(user_msg, expanded_keywords, top_k=15)

        # BƯỚC 3: PROMPT ENGINEERING (Chain-of-Thought)
        # Bắt buộc AI phân tích trước khi code để tránh bịa đặt
        system_prompt = f"""Role: Senior BigQuery SQL Architect.
Goal: Generate optimized Standard SQL queries based strictly on the provided schema.

[CONTEXT - DATABASE SCHEMAS]:
{relevant_schemas}

[INSTRUCTIONS - READ CAREFULLY]:
1. **Schema Check (CRITICAL):** - You MUST first identify which tables from the [CONTEXT] match the user's request.
   - **DO NOT** use any table name or column that is not explicitly listed in [CONTEXT].
   - If a table is missing, tell the user you cannot find it. DO NOT GUESS.

2. **Table Selection:**
   - Prefer `Type: BASE TABLE` over `Type: VIEW` if both exist for the same data.
   - Always use the Full Qualified Name (e.g., `project.dataset.table`) from the [TABLE] header.

3. **Function/Routine Logic:**
   - Check `[FUNCTION]` definitions. If a user asks for a specific status/condition, look at the Routine SQL `CASE WHEN` to find the corresponding ID.
   - Use the ID in the `WHERE` clause.

[RESPONSE FORMAT]:
You must structure your response exactly like this:

**1. Analysis:**
* **Tables identified:** [List exact table names found in CONTEXT]
* **Columns selected:** [List exact columns]
* **Logic:** [Explain how you map text to IDs using Functions if any]

**2. SQL Query:**
```sql
-- Your Standard SQL here
User Question: {user_msg} """
        messages_payload = [{"role": "system", "content": system_prompt}]
        history = get_chat_history_formatted(session_id, limit=4) 
        
        for msg in history:
            if msg['content'] != user_msg:
                messages_payload.append(msg)
        
        messages_payload.append({"role": "user", "content": user_msg})

        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        
        # GỌI OLLAMA
        response = client.chat(
            model=MODEL_NAME,
            messages=messages_payload,
            stream=False,
            # QUAN TRỌNG: Temperature = 0.0 để loại bỏ sự sáng tạo/bịa đặt
            options={"temperature": 0.0} 
        )
        
        reply = response['message']['content']
        save_message(session_id, "assistant", reply)
        return jsonify({"response": reply})

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__': # Init DB & RAG khi chạy local init_db() rag_engine.load_schemas() 
    app.run(debug=True, port=5000)



