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
        Tokenizer đơn giản hóa để tăng khả năng bắt từ (Recall).
        Giữ lại cả từ đơn và cụm từ.
        """
        text = str(text).lower()
        # Thay thế ký tự đặc biệt bằng khoảng trắng
        text = re.sub(r'[\_\-\.\,]', ' ', text)
        
        tokens = text.split()
        
        # Stopwords tối thiểu thôi, đừng lọc 'table' hay 'date' vì đôi khi user hỏi đích danh
        stopwords = {'select', 'from', 'where', 'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of'}
        
        final_tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        
        # Thêm biến thể n-gram đơn giản (Optional but good for 'user id')
        # Nhưng để BM25 hoạt động tốt nhất với Schema, ta giữ token đơn là chính.
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
        Dùng AI để dịch và mở rộng từ khóa.
        QUAN TRỌNG: Phải dịch từ tiếng Việt sang tiếng Anh kỹ thuật (Database Terms).
        """
        try:
            client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
            prompt = f"""You are a Database Expert.
User Query: "{user_query}"

Task: Convert this query into a list of Technical SQL Keywords (English).
1. Translate Vietnamese terms to English (e.g., 'học viên' -> 'student user learner', 'đơn hàng' -> 'order transaction').
2. Identify potential table names (snake_case) and columns.
3. OUTPUT ONLY the list of keywords separated by spaces. Do NOT explain.
"""
            
            response = client.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0}
            )
            keywords = response['message']['content']
            # Clean up: chỉ lấy chữ cái và số, bỏ ký tự lạ
            keywords = re.sub(r'[^\w\s]', '', keywords)
            print(f"🔹 Expanded Query: {keywords}")
            return keywords
        except Exception as e:
            print(f"Expansion Error: {e}")
            return user_query

    def retrieve(self, query, expanded_query=None, top_k=20): # Tăng top_k lên 20
        if not self.is_ready: return ""
        
        # Kết hợp: Query gốc (VN) + Expanded (EN)
        search_query = f"{query} {expanded_query}"
        tokenized_query = self.tokenize(search_query)
        
        # 1. Raw Scores từ BM25
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. Lấy Top 50 ứng viên
        top_n_candidates = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:50]
        
        final_scored_candidates = []
        query_lower = query.lower()
        expanded_lower = str(expanded_query).lower()
        
        for idx in top_n_candidates:
            # Cho phép điểm 0 nếu keyword match (đôi khi BM25 tính điểm gắt)
            original_score = doc_scores[idx]
            
            metadata = self.doc_map.get(idx, {})
            short_name = metadata.get('short_name', '')
            is_view = metadata.get('is_view', False)
            obj_type = metadata.get('type')
            
            boost = 0.0
            
            # Rule A: Name Match (Check cả trong query gốc lẫn expanded query)
            # Ví dụ: Expanded ra 'booking' -> Match bảng 'bookings'
            if short_name:
                # Check trong query tiếng Việt
                if short_name in query_lower: 
                    boost += 20.0
                # Check trong query tiếng Anh (expanded)
                elif short_name in expanded_lower:
                    boost += 15.0
            
            # Rule B: Ưu tiên Table
            if obj_type == 'table' and not is_view:
                boost += 5.0 
            elif is_view:
                boost -= 1.0 # Giảm nhẹ thôi
                
            final_score = original_score + boost
            
            # Chỉ lấy nếu có điểm > 0 hoặc boost > 0
            if final_score > 0:
                final_scored_candidates.append((idx, final_score))
            
        # 3. Sort và lấy Top K
        final_scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        results = [self.schema_docs[idx] for idx, score in final_scored_candidates[:top_k]]
        
        if not results:
            print("⚠️ Không tìm thấy schema, thử lấy top 5 mặc định.")
            return "\n".join(self.schema_docs[:5])

        # [DEBUG] In ra tên các bảng tìm được để kiểm tra
        print(f"✅ RAG Retrieved {len(results)} tables.") 
        # (Bạn có thể bỏ dòng print này khi chạy prod)
            
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
if __name__ == 'main': # Init DB & RAG khi chạy local init_db() rag_engine.load_schemas() 
    app.run(debug=True, port=5000)
