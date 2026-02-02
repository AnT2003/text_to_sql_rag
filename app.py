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
MODEL_NAME = "gemini-3-flash-preview:latest"
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
#  PHẦN 3: ADVANCED RAG ENGINE (CORE LOGIC)
# =========================================================

class RAGEngine:
    def __init__(self):
        self.schema_docs = [] # Lưu nội dung text full
        self.bm25 = None      # Object tìm kiếm BM25
        self.doc_map = {}     # Map index -> doc data
        self.is_ready = False

    def tokenize(self, text):
        """
        Kỹ thuật Tokenization chuyên cho SQL:
        - Tách snake_case (user_id -> user, id)
        - Tách camelCase
        - Loại bỏ từ thừa (stopwords)
        """
        # Chuyển các ký tự đặc biệt thành dấu cách
        text = re.sub(r'[\.\_\-\(\)\,]', ' ', str(text))
        # Tách camelCase (e.g., camelCase -> camel Case)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        tokens = text.lower().split()
        
        # Stopwords SQL thông dụng không mang ý nghĩa định danh
        stopwords = {
            'string', 'int64', 'float', 'boolean', 'timestamp', 'date', 
            'table', 'dataset', 'project', 'nullable', 'mode', 'type', 
            'description', 'record', 'create', 'replace'
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def load_schemas(self):
        print("🚀 Đang khởi tạo Advanced RAG Indexing...")
        new_docs = []
        tokenized_corpus = []
        
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
                        # ===============================
                        # Build Document Content
                        # ===============================
                        doc_content = ""
                        keywords_source = ""
                        if 'table_name' in item:
                            table_name = item.get("table_name", "")
                            ddl = item.get("ddl", "")

                            dataset_id = item.get('table_schema')
                            project_id = 'kynaforkids-server-production'

                            # Prefix đầy đủ: `project.dataset` hoặc `dataset`
                            full_prefix = f"{project_id}.{dataset_id}" if project_id else dataset_id
                            
                            table_name = item['table_name']
                            full_table_name = f"`{full_prefix}.{table_name}`"

                            # -------------------------------
                            # Parse columns (JSON string)
                            # -------------------------------
                            cols_desc = []
                            col_tokens = []

                            raw_cols = item.get("columns", "[]")
                            try:
                                parsed_cols = json.loads(raw_cols) if isinstance(raw_cols, str) else raw_cols
                                if isinstance(parsed_cols, list):
                                    for col in parsed_cols:
                                        cols_desc.append(f"- {col}")
                                        col_tokens.append(col)
                            except Exception:
                                pass

                            # -------------------------------
                            # Build doc_content
                            # -------------------------------
                            doc_content = (
                                f"[TABLE] {full_table_name}\n"
                                f"Type: {item.get('table_type', '')}\n"
                                f"Columns:\n" + "\n".join(cols_desc)
                            )

                            # -------------------------------
                            # Build keywords source (for index)
                            # -------------------------------
                            keywords_source = f"{full_table_name} " + " ".join(col_tokens)


                        elif 'routine_name' in item:
                            r_name = item['routine_name']
                            short_name = item.get('routine_name')
                            ddl = item.get('ddl', '')
                            definition = item.get('routine_definition', '')

                            # Regex bắt tên function
                            match = re.search(r'FUNCTION\s+`([^`]+)`', ddl, re.IGNORECASE)
                            full_name = f"`{match.group(1)}`" if match else f"`{short_name}`"
                            definition = item.get('routine_definition', '')
                            doc_content = f"[FUNCTION] {full_name}\nLogic:\n{definition}"
                            keywords_source = f"{full_name} {definition}"

                        if doc_content:
                            new_docs.append(doc_content)
                            tokenized_corpus.append(self.tokenize(keywords_source))

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # KHỞI TẠO BM25 INDEX
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.schema_docs = new_docs
            self.is_ready = True
            print(f"✅ Đã index {len(new_docs)} schemas với BM25.")
        else:
            print("⚠️ Không có dữ liệu để index.")

    def query_expansion(self, user_query, api_key):
        """
        Kỹ thuật 'Query Expansion': Dùng AI nhỏ để dịch câu hỏi người dùng
        sang các từ khóa Database tiềm năng trước khi search.
        Ví dụ: "Tổng tiền bán hàng" -> "revenue sales total amount transaction"
        """
        try:
            client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
            prompt = f"""You are a SQL search assistant. 
            User Query: "{user_query}"
            Task: List 5-10 technical database keywords (in English) related to this query.
            Focus on synonyms for table names or column names (e.g., if user says "client", output "customer user account profile").
            Output only the keywords separated by spaces. No explanation."""
            
            response = client.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0}
            )
            keywords = response['message']['content']
            print(f"🔹 Expanded Query: {keywords}")
            return keywords
        except:
            return user_query # Fallback nếu lỗi

    def retrieve(self, query, expanded_query=None, top_k=10):
        if not self.is_ready: return ""
        
        # Kết hợp query gốc và query mở rộng để tìm kiếm toàn diện
        search_query = f"{query} {expanded_query}" if expanded_query else query
        tokenized_query = self.tokenize(search_query)
        
        # BM25 Scoring
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Lấy top K indices
        top_n = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        # Filter: Chỉ lấy những doc có score > 0 (tránh lấy rác nếu không khớp gì)
        results = [self.schema_docs[i] for i in top_n if doc_scores[i] > 0]
        
        if not results:
            print("⚠️ Không tìm thấy schema khớp, lấy default top 2.")
            return "\n".join(self.schema_docs[:2])
            
        return "\n--------------------\n".join(results)

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
        relevant_schemas = rag_engine.retrieve(user_msg, expanded_keywords, top_k=10)

        # BƯỚC 3: PROMPT ENGINEERING (Context-Aware Generation)
        system_prompt = f"""Role: Senior BigQuery SQL Architect.
Goal: Generate optimized Standard SQL queries based strictly on the provided schema.

[CONTEXT - RELEVANT SCHEMAS]:
{relevant_schemas}

[GUIDELINES]:
1. **Source of Truth**: Use ONLY the tables/columns provided in [CONTEXT]. Do not hallucinate columns.
2. **Expansion Context**: The user query might use business terms. Map them to the technical column names found in the schema.
3. **Logic Handling**: If a [FUNCTION] or Routine is present in context, use its logic (CASE WHEN...) to filter data correctly (e.g., status codes). You MUST reuse its logic exactly as defined. Do NOT re-implement, simplify, or invent CASE WHEN logic.
4. **Syntax**: Use Google Standard SQL (BigQuery) syntax. usage of backticks (`) for table names is mandatory (Project.Dataset.Table).
5. **Output**: Return ONLY the SQL code inside ```sql ... ``` block. Brief explanation of the query is optional after the code block.

User Question: {user_msg}
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
    # Init DB & RAG trước khi run app
    init_db()
    rag_engine.load_schemas()
    app.run(debug=True, port=5000)
