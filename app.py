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
#  PHẦN 1: CONFIG & SETUP MÔI TRƯỜNG
# =========================================================

load_dotenv()
app = Flask(__name__)
CORS(app)

# Cấu hình Database (Tự động thích ứng SQLite/Postgres cho Render/Local)
db_url = os.getenv("DATABASE_URL")
if not db_url:
    # Mặc định dùng SQLite nếu chạy local
    db_url = "sqlite:///local_chat.db"
# Fix lỗi protocol của Postgres trên Render (nếu có)
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Cấu hình AI Ollama
OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gemini-3-flash-preview:latest" # Thay đổi model tùy setup của bạn
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY")
SCHEMA_FOLDER = "./schemas"

# BIẾN TOÀN CỤC: Lưu trữ bộ nhớ Schemas trên RAM
SCHEMA_DOCS = []        # List chứa nội dung text clean để gửi AI
BM25_MODEL = None       # Model tìm kiếm

# =========================================================
#  PHẦN 2: DATABASE MODELS (LƯU LỊCH SỬ CHAT)
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

# =========================================================
#  PHẦN 3: HÀM HỖ TRỢ DATABASE
# =========================================================

def init_db():
    with app.app_context():
        db.create_all()
        print("✅ Database Connected.")

def save_message(session_id, role, content):
    try:
        new_msg = Message(session_id=session_id, role=role, content=content)
        db.session.add(new_msg)
        db.session.commit()
    except Exception as e:
        print(f"Error saving message: {e}")
        db.session.rollback()

def create_session_if_not_exists(session_id, first_msg):
    try:
        if not Session.query.get(session_id):
            title = (first_msg[:50] + '...') if len(first_msg) > 50 else first_msg
            db.session.add(Session(id=session_id, title=title))
            db.session.commit()
    except Exception as e:
        print(f"Error creating session: {e}")
        db.session.rollback()

def get_chat_history_formatted(session_id, limit=10):
    try:
        msgs = Message.query.filter_by(session_id=session_id).order_by(desc(Message.created_at)).limit(limit).all()
        return [{"role": m.role, "content": m.content} for m in msgs[::-1]]
    except:
        return []

# =========================================================
#  PHẦN 4: LOGIC LOAD SCHEMA & RAG ENGINE (CORE)
# =========================================================

def simple_tokenizer(text):
    """
    Tokenizer tối ưu cho SQL:
    - Giữ lại dấu gạch dưới (_) để tìm tên bảng chính xác (vd: Acc_LTV).
    - Tách camelCase (vd: CacCost -> Cac Cost).
    """
    text = str(text)
    # Tách camelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Thay thế ký tự lạ bằng khoảng trắng, nhưng giữ lại chữ số, chữ cái và gạch dưới
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.lower().split()
    stopwords = {'create', 'table', 'view', 'external', 'float64', 'string', 'date', 'int64', 'struct', 'array', 'replace', 'exists', 'options', 'sheets'}
    return [t for t in tokens if t not in stopwords]

def load_all_schemas():
    """
    Đọc JSON, dùng Regex trích xuất tên bảng từ DDL và xây dựng Index BM25.
    """
    global SCHEMA_DOCS, BM25_MODEL
    print("🚀 Đang nạp và Index Schemas (Accuracy Mode)...")

    if not os.path.exists(SCHEMA_FOLDER):
        print(f"⚠️ Không tìm thấy thư mục {SCHEMA_FOLDER}")
        return

    json_files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))
    schema_parts = []
    tokenized_corpus = []

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]

                for item in items:
                    # -------------------------------------------------------
                    # 1. XỬ LÝ TABLE / VIEW
                    # -------------------------------------------------------
                    if 'table_name' in item and 'ddl' in item:
                        ddl = item['ddl']
                        short_name = item['table_name']
                        
                        # --- CẢI TIẾN REGEX ---
                        # Pattern này tìm cụm `TABLE `tên_bảng`` bất kể prefix (CREATE OR REPLACE...)
                        match = re.search(r'(?:TABLE|VIEW)\s+`([^`]+)`', ddl, re.IGNORECASE)
                        
                        if match:
                            full_table_name = f"`{match.group(1)}`" # output: `project.dataset.table`
                        else:
                            # Fallback an toàn: cố gắng tìm chuỗi có dạng a.b.c trong toàn bộ DDL
                            match_loose = re.search(r'`([\w\-]+\.[\w\-]+\.[\w\-]+)`', ddl)
                            full_table_name = match_loose.group(0) if match_loose else f"`UnknownProject.UnknownDataset.{short_name}`"

                        table_type = item.get('table_type', 'TABLE')
                        
                        cols = []
                        col_tokens = []
                        raw_columns = item.get('columns')
                        
                        if raw_columns:
                            try:
                                parsed = json.loads(raw_columns) if isinstance(raw_columns, str) else raw_columns
                                if isinstance(parsed, list):
                                    for col in parsed:
                                        c_name = col if isinstance(col, str) else col.get('name')
                                        cols.append(f"- {c_name}")
                                        col_tokens.append(c_name)
                            except: pass
                        
                        # Nội dung Clean để AI đọc
                        content_block = f"""
[TABLE SCHEMA]
ID: {full_table_name}
Short Name: {short_name}
Type: {table_type}
Columns:
{chr(10).join(cols)}
Source DDL:
```sql
{ddl}
```
"""
                        # Tối ưu Search Text: Lặp lại tên bảng để tăng trọng số (Weighting)
                        # Khi user search tên bảng, điểm BM25 sẽ rất cao
                        search_text = f"{full_table_name} {short_name} {short_name} {' '.join(col_tokens)}"
                        
                        schema_parts.append({"text": content_block, "search_text": search_text})
                        tokenized_corpus.append(simple_tokenizer(search_text))

                    # -------------------------------------------------------
                    # 2. XỬ LÝ ROUTINE / FUNCTION
                    # -------------------------------------------------------
                    elif 'routine_name' in item:
                        short_name = item.get('routine_name')
                        ddl = item.get('ddl', '')
                        definition = item.get('routine_definition', '')

                        # Regex bắt tên function
                        match = re.search(r'FUNCTION\s+`([^`]+)`', ddl, re.IGNORECASE)
                        full_name = f"`{match.group(1)}`" if match else f"`{short_name}`"

                        content_block = f"""
[FUNCTION SCHEMA]
ID: {full_name}
Logic Body:
{definition}
"""
                        search_text = f"{full_name} {short_name} {short_name} {definition}"
                        schema_parts.append({"text": content_block, "search_text": search_text})
                        tokenized_corpus.append(simple_tokenizer(search_text))

        except Exception as e:
            print(f"❌ Lỗi file {file_path}: {e}")

    SCHEMA_DOCS = schema_parts
    if tokenized_corpus:
        BM25_MODEL = BM25Okapi(tokenized_corpus)
        print(f"✅ Đã index {len(SCHEMA_DOCS)} schemas thành công.")
    else:
        print("⚠️ Không có schema nào được nạp.")

def retrieve_schema_smart(question, top_k=6):
    """
    Tìm kiếm schema liên quan nhất dựa trên BM25.
    Lấy Top 6 để đảm bảo đủ context nhưng không thừa thãi.
    """
    if not BM25_MODEL or not SCHEMA_DOCS: return ""
    
    tokens = simple_tokenizer(question)
    doc_scores = BM25_MODEL.get_scores(tokens)
    
    # Sắp xếp index theo điểm số giảm dần
    top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
    
    # Chỉ lấy kết quả có độ tương đồng > 0 (loại bỏ rác)
    results = [SCHEMA_DOCS[i]['text'] for i in top_indices if doc_scores[i] > 0]
    
    # Fallback: Nếu không tìm thấy gì, lấy 2 bảng đầu tiên để AI không bị blank
    if not results and SCHEMA_DOCS:
        results = [d['text'] for d in SCHEMA_DOCS[:2]]
        
    return "\n--------------------\n".join(results)

# --- Khởi chạy nạp dữ liệu khi Start App ---
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
    try:
        sessions = Session.query.order_by(desc(Session.created_at)).all()
        return jsonify([{'id': s.id, 'title': s.title} for s in sessions])
    except: return jsonify([])

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
    except: return jsonify({"error": "Failed to clear history"}), 500

@app.route("/api/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    try:
        Message.query.filter_by(session_id=session_id).delete()
        Session.query.filter_by(id=session_id).delete()
        db.session.commit()
        return jsonify({"status": "success"})
    except: return jsonify({"error": "Failed to delete session"}), 500

@app.route('/api/reload', methods=['POST'])
def reload_schema_api():
    load_all_schemas()
    return jsonify({"status": "success", "message": "Schemas reloaded & re-indexed!"})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    api_key = data.get('api_key') or DEFAULT_API_KEY
    user_msg = data.get('message')
    session_id = data.get('session_id')

    if not api_key or not session_id:
        return jsonify({"error": "Missing API Key or Session ID"}), 400

    # 1. RETRIEVAL (Lấy context thông minh)
    retrieved_context = retrieve_schema_smart(user_msg, top_k=6)

    create_session_if_not_exists(session_id, user_msg)
    save_message(session_id, "user", user_msg)

    # 2. PROMPT ENGINEERING (Siết chặt quy tắc để tránh bịa đặt)
    system_prompt = f"""Bạn là một chuyên gia SQL BigQuery.
Nhiệm vụ: Viết câu lệnh SQL Standard dựa trên yêu cầu người dùng và Schema được cung cấp.

[CONTEXT - DATABASE SCHEMA]:
{retrieved_context}

[QUY TẮC BẤT DI BẤT DỊCH - PHẢI TUÂN THỦ]:
1. **ĐỊNH DANH BẢNG (QUAN TRỌNG NHẤT)**:
   - Bạn PHẢI sử dụng tên bảng đầy đủ (Full Qualified Name) được ghi tại dòng `ID:` trong [CONTEXT].
   - Ví dụ: Nếu Context ghi `ID: `kyna.data.users``, bạn phải viết `FROM `kyna.data.users``.
   - TUYỆT ĐỐI KHÔNG dùng tên viết tắt (vd: `..users`), không tự ý bịa Project ID nếu Context không có.
   
2. **SỰ THẬT**:
   - Chỉ sử dụng các bảng và cột CÓ TRONG CONTEXT.
   - Nếu không tìm thấy bảng phù hợp, hãy trả lời: "Xin lỗi, tôi không tìm thấy thông tin bảng liên quan trong dữ liệu hiện có."

3. **LOGIC**:
   - Đọc kỹ [FUNCTION SCHEMA] (nếu có) để hiểu logic tính toán (ví dụ: status=1 nghĩa là gì).
   - Sử dụng cú pháp Google Standard SQL (BigQuery).

User Question: {user_msg}

[OUTPUT FORMAT]:
Chỉ trả về code SQL trong ```sql ... ```. Kèm giải thích ngắn gọn.
"""

    # Xây dựng message payload
    messages_payload = [{"role": "system", "content": system_prompt}]
    history = get_chat_history_formatted(session_id, limit=6)
    for msg in history:
        if msg['content'] != user_msg:
            messages_payload.append(msg)
    messages_payload.append({"role": "user", "content": user_msg})

    try:
        # Gọi API Ollama
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        response = client.chat(
            model=MODEL_NAME,
            messages=messages_payload,
            stream=False,
            options={"temperature": 0.0} # Nhiệt độ = 0 để tối đa hóa tính chính xác logic
        )
        reply = response['message']['content']
        save_message(session_id, "assistant", reply)
        return jsonify({"response": reply})

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
