import os
import json
import glob
import datetime
import re  # <--- Bắt buộc có để trích xuất tên từ DDL
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from rank_bm25 import BM25Okapi  # Thư viện RAG tối ưu

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

# BIẾN TOÀN CỤC
SCHEMA_DOCS = []        # List chứa các đoạn text schema
BM25_MODEL = None       # Model tìm kiếm

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
#  PHẦN 3: LOGIC LOAD SCHEMA (SỬ DỤNG REGEX TRÍCH XUẤT TỪ DDL)
# =========================================================

def load_all_schemas():
    """
    Load schema từ JSON. Vì JSON không có tableReference, ta dùng Regex 
    để 'bóc' tên bảng đầy đủ từ chuỗi DDL.
    """
    global SCHEMA_DOCS
    print("🚀 Đang nạp và xử lý DDL từ Schemas...")

    if not os.path.exists(SCHEMA_FOLDER):
        print(f"⚠️ Không tìm thấy thư mục {SCHEMA_FOLDER}")
        return

    json_files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))
    schema_parts = []

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]

                for item in items:
                    # 1. XỬ LÝ TABLE / VIEW
                    if 'table_name' in item and 'ddl' in item:
                        ddl = item['ddl']
                        table_name_short = item['table_name']
                        
                        # --- MAGIC REGEX ---
                        # Tìm chuỗi nằm giữa dấu backtick (`) sau chữ TABLE hoặc VIEW
                        # Pattern này bắt được: CREATE EXTERNAL TABLE `a.b.c` hoặc CREATE VIEW `a.b.c`
                        match = re.search(r'CREATE.*?(?:TABLE|VIEW)\s+`([^`]+)`', ddl, re.IGNORECASE | re.DOTALL)
                        
                        if match:
                            # Lấy được: kynaforkids-server-production.kynaforkids.Acc_LTV_CAC
                            full_table_name = f"`{match.group(1)}`"
                        else:
                            # Fallback nếu DDL dị biệt (ít xảy ra với BigQuery export)
                            full_table_name = f"`{table_name_short}`"

                        table_type = item.get('table_type', 'TABLE')
                        
                        cols = []
                        raw_columns = item.get('columns')
                        col_tokens = [] # Dùng để đánh index tìm kiếm
                        
                        if raw_columns:
                            try:
                                parsed_columns = json.loads(raw_columns)
                                if isinstance(parsed_columns, list):
                                    for col in parsed_columns:
                                        # Xử lý trường hợp col là string hoặc dict
                                        c_name = col if isinstance(col, str) else col.get('name')
                                        cols.append(f"- `{c_name}`")
                                        col_tokens.append(c_name)
                            except: pass
                        
                        columns_block = "\n".join(cols)
                        
                        # Nội dung để AI đọc
                        content_block = f"""
                        [TABLE ENTITY]
                        Table Name: {full_table_name}
                        Table Type: {table_type}
                        Source DDL:
                        ```sql
                        {ddl}
                        ```
                        COLUMNS:
                        {columns_block}
                        """
                        
                        # Dữ liệu để RAG đánh index (Full name + short name + columns)
                        # clean_text giúp BM25 hiểu được các từ dính nhau bằng dấu chấm
                        search_text = f"{full_table_name.replace('.', ' ')} {table_name_short} {' '.join(col_tokens)}"
                        
                        schema_parts.append({"text": content_block, "search_text": search_text})

                    # 2. XỬ LÝ ROUTINE / FUNCTION
                    elif 'routine_name' in item:
                        # Tương tự, nếu Routine có DDL thì trích xuất, nếu không thì tự ghép
                        routine_name = item.get('routine_name')
                        ddl = item.get('ddl', '')
                        definition = item.get('routine_definition', '')
                        
                        # Regex tìm tên function trong DDL
                        match = re.search(r'CREATE.*?FUNCTION\s+`([^`]+)`', ddl, re.IGNORECASE | re.DOTALL)
                        if match:
                             full_routine_name = f"`{match.group(1)}`"
                        else:
                             full_routine_name = f"`{routine_name}`"

                        content_block = f"""
                        [LOGIC ROUTINE]
                        Routine Name: {full_routine_name}
                        Definition:
                        {definition}
                        """
                        search_text = f"{full_routine_name.replace('.', ' ')} {definition}"
                        schema_parts.append({"text": content_block, "search_text": search_text})

        except Exception as e:
            print(f"❌ Lỗi file {file_path}: {e}")

    SCHEMA_DOCS = schema_parts
    # Xây dựng Index ngay
    build_rag_index()

# =========================================================
#  PHẦN 4: RAG ENGINE (BM25)
# =========================================================

def tokenize_query(text):
    """Tách từ: xóa ký tự đặc biệt, tách camelCase, xóa stopword"""
    text = re.sub(r'[\.\_\-\(\)\,\`]', ' ', str(text))
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    tokens = text.lower().split()
    stopwords = {'create', 'table', 'view', 'select', 'external', 'float64', 'string', 'date'}
    return [t for t in tokens if t not in stopwords]

def build_rag_index():
    global BM25_MODEL, SCHEMA_DOCS
    if not SCHEMA_DOCS: return
    
    # Tokenize field 'search_text' ta đã chuẩn bị ở trên
    tokenized_corpus = [tokenize_query(doc['search_text']) for doc in SCHEMA_DOCS]
    BM25_MODEL = BM25Okapi(tokenized_corpus)
    print(f"✅ Đã index {len(SCHEMA_DOCS)} schemas thành công.")

def retrieve_schema_smart(question, top_k=5):
    if not BM25_MODEL or not SCHEMA_DOCS: return ""
    
    tokenized_query = tokenize_query(question)
    doc_scores = BM25_MODEL.get_scores(tokenized_query)
    
    # Lấy top k index có điểm cao nhất
    top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
    
    # Lọc bỏ những kết quả điểm = 0 (không liên quan tí nào)
    results = [SCHEMA_DOCS[i]['text'] for i in top_indices if doc_scores[i] > 0]
    
    # Fallback: Nếu không tìm thấy gì, lấy đại 2 cái đầu (để AI không bị blank context)
    if not results:
        results = [d['text'] for d in SCHEMA_DOCS[:2]]
        
    return "\n--------------------\n".join(results)

# Khởi tạo
init_db()
load_all_schemas()

# =========================================================
#  PHẦN 5: API ROUTES
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
    except: return jsonify({"error": "err"}), 500

@app.route("/api/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    try:
        Message.query.filter_by(session_id=session_id).delete()
        Session.query.filter_by(id=session_id).delete()
        db.session.commit()
        return jsonify({"status": "success"})
    except: return jsonify({"error": "err"}), 500

@app.route('/api/reload', methods=['POST'])
def reload_schema_api():
    load_all_schemas()
    return jsonify({"status": "success", "message": "Schemas reloaded!"})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    api_key = data.get('api_key') or DEFAULT_API_KEY
    user_msg = data.get('message')
    session_id = data.get('session_id')

    if not api_key or not session_id:
        return jsonify({"error": "Missing info"}), 400

    # 1. RETRIEVAL (BM25)
    retrieved_context = retrieve_schema_smart(user_msg, top_k=5)

    create_session_if_not_exists(session_id, user_msg)
    save_message(session_id, "user", user_msg)

    # 2. PROMPT
    # Lưu ý: Phần Prompt này nhấn mạnh việc COPY tên bảng từ context
    system_prompt = f"""Role: BigQuery SQL Expert.
Nhiệm vụ: Chuyển câu hỏi người dùng thành câu lệnh SQL Standard.

[DATABASE SCHEMA - RELEVANT CONTEXT]:
{retrieved_context}

[QUY TẮC BẮT BUỘC]:
1. **FULL NAME ONLY**: Phải dùng tên bảng đầy đủ CHÍNH XÁC như trong phần 'Table Name:' ở trên (ví dụ: `project.dataset.table`).
   - Tuyệt đối KHÔNG dùng tên viết tắt kiểu `..table` hay `.table`.
   - Nếu trong schema ghi `UnknownDataset.table`, hãy dùng y nguyên `UnknownDataset.table`.
2. **Logic Mapping**: Đọc kỹ phần [LOGIC ROUTINE] (nếu có) để map các trạng thái (status, type) sang số/mã tương ứng trong WHERE clause.
3. **Syntax**: Dùng Google Standard SQL.

User Question: {user_msg}

[ĐỊNH DẠNG TRẢ VỀ]:
Chỉ trả về code SQL trong ```sql ... ```. Sau khi trả kết quả SQL nên có thêm phần giải thích ngắn gọn.
"""

    messages_payload = [{"role": "system", "content": system_prompt}]
    history = get_chat_history_formatted(session_id, limit=6)
    for msg in history:
        if msg['content'] != user_msg:
            messages_payload.append(msg)
    messages_payload.append({"role": "user", "content": user_msg})

    try:
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        response = client.chat(
            model=MODEL_NAME,
            messages=messages_payload,
            stream=False,
            options={"temperature": 0.0}
        )
        reply = response['message']['content']
        save_message(session_id, "assistant", reply)
        return jsonify({"response": reply})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    load_all_schemas()
    app.run(debug=True, port=5000)
