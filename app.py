import os
import json
import glob
import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc

# --- IMPORT THƯ VIỆN RAG NÂNG CAO ---
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- 1. SETUP & CẤU HÌNH ---
load_dotenv()
app = Flask(__name__)
CORS(app)

SCHEMA_FOLDER = "./schemas"
OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gpt-oss:120b"
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY")

# Cấu hình Database (Tự động thích ứng SQLite/Postgres)
db_url = os.getenv("DATABASE_URL", "sqlite:///chat_history.db")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# BIẾN TOÀN CỤC (CACHE)
# 1. Chứa toàn bộ Logic Routines (Luôn gửi cho AI)
GLOBAL_ROUTINES_CONTEXT = "" 
# 2. Bộ tìm kiếm Bảng (Chỉ tìm bảng liên quan)
ENSEMBLE_RETRIEVER = None

# =========================================================
#  PHẦN 2: QUẢN LÝ DATABASE (LƯU LỊCH SỬ)
# =========================================================
class Session(db.Model):
    __tablename__ = 'sessions'
    id = db.Column(db.String(50), primary_key=True)
    title = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), db.ForeignKey('sessions.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

def init_db():
    with app.app_context():
        db.create_all()
        print("✅ Database Connected.")

def save_message(session_id, role, content):
    try:
        new_msg = Message(session_id=session_id, role=role, content=content)
        db.session.add(new_msg)
        db.session.commit()
    except: db.session.rollback()

def create_session_if_not_exists(session_id, first_msg):
    try:
        session = Session.query.get(session_id)
        if not session:
            title = (first_msg[:50] + '...') if len(first_msg) > 50 else first_msg
            db.session.add(Session(id=session_id, title=title))
            db.session.commit()
    except: db.session.rollback()

def get_chat_history_formatted(session_id, limit=10):
    msgs = Message.query.filter_by(session_id=session_id).order_by(desc(Message.created_at)).limit(limit).all()
    history = []
    for m in msgs[::-1]: history.append({"role": m.role, "content": m.content})
    return history

# =========================================================
#  PHẦN 3: ADVANCED RAG INITIALIZATION
# =========================================================
def init_advanced_rag():
    """
    Khởi tạo hệ thống RAG phân tầng:
    1. Routines: Nạp Full vào biến toàn cục (High Priority).
    2. Tables: Index vào Vector Store & BM25 (Retrieval Priority).
    """
    global GLOBAL_ROUTINES_CONTEXT, ENSEMBLE_RETRIEVER
    print("🚀 Đang khởi tạo Advanced RAG System...")

    if not os.path.exists(SCHEMA_FOLDER): return

    json_files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))
    
    table_docs = []
    routine_texts = []

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    # --- XỬ LÝ ROUTINE (Hàm Logic) ---
                    # Logic: Hàm chứa quy tắc nghiệp vụ (CASE WHEN), AI cần thấy nó MỌI LÚC.
                    if 'routine_name' in item:
                        name = item.get('routine_name', 'Unknown')
                        definition = item.get('routine_definition') or item.get('ddl') or ''
                        # Tạo đoạn văn bản mô tả routine
                        r_text = f"FUNCTION: {name}\nLOGIC:\n```sql\n{definition}\n```"
                        routine_texts.append(r_text)

                    # --- XỬ LÝ TABLE (Bảng Dữ liệu) ---
                    # Logic: Bảng rất nhiều, chỉ tìm bảng liên quan khi cần.
                    elif 'table_name' in item:
                        name = item.get('table_name', 'Unknown')
                        desc = item.get('description', '')
                        cols = [f"{c['name']} ({c.get('type')})" for c in item.get('columns', [])]
                        col_str = ", ".join(cols) # Gộp gọn để tiết kiệm token
                        
                        # Nội dung để Index (Tìm kiếm)
                        page_content = f"TABLE: {name}\nDESC: {desc}\nCOLS: {col_str}\nFULL_SCHEMA: {json.dumps(item, ensure_ascii=False)}"
                        
                        table_docs.append(Document(page_content=page_content, metadata={"source": name}))

        except Exception as e:
            print(f"❌ Lỗi file {file_path}: {e}")

    # 1. Lưu Routines vào Global Context
    if routine_texts:
        GLOBAL_ROUTINES_CONTEXT = "\n====================\n".join(routine_texts)
        print(f"✅ Đã nạp {len(routine_texts)} Routines vào Global Memory.")
    else:
        GLOBAL_ROUTINES_CONTEXT = "No routines found."

    # 2. Tạo Bộ tìm kiếm Tables (Hybrid: Semantic + Keyword)
    if table_docs:
        print("⏳ Đang tạo Table Index...")
        # Vector Search
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(table_docs, embeddings)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Keyword Search
        bm25_retriever = BM25Retriever.from_documents(table_docs)
        bm25_retriever.k = 5
        
        # Ensemble (Kết hợp)
        ENSEMBLE_RETRIEVER = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        print(f"✅ Table RAG sẵn sàng ({len(table_docs)} bảng).")

def retrieve_tables(query):
    """Tìm bảng liên quan bằng Hybrid Search"""
    if not ENSEMBLE_RETRIEVER: return ""
    docs = ENSEMBLE_RETRIEVER.invoke(query)
    # Deduplicate (loại bỏ trùng)
    seen = set()
    unique_docs = []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)
    
    return "\n---\n".join([d.page_content for d in unique_docs[:6]])

# --- KHỞI CHẠY ---
init_db()
init_advanced_rag()

# =========================================================
#  PHẦN 4: API ROUTES & PROMPT ENGINEERING
# =========================================================

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    sessions = Session.query.order_by(desc(Session.created_at)).all()
    return jsonify([{'id': s.id, 'title': s.title, 'created_at': s.created_at} for s in sessions])

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    return jsonify(get_chat_history_formatted(session_id, limit=50))

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    try:
        Message.query.delete()
        Session.query.delete()
        db.session.commit()
        return jsonify({"status": "success", "message": "Deleted all history."})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    api_key = data.get('api_key') or DEFAULT_API_KEY
    user_msg = data.get('message')
    session_id = data.get('session_id')

    if not api_key or not session_id: return jsonify({"error": "Thiếu thông tin"}), 400

    try:
        create_session_if_not_exists(session_id, user_msg)
        save_message(session_id, "user", user_msg)

        # 1. LẤY CONTEXT (Kỹ thuật Advanced: Global Logic + Retrieved Data)
        # Luôn lấy toàn bộ Logic hàm
        logic_context = GLOBAL_ROUTINES_CONTEXT
        # Chỉ lấy Bảng liên quan
        data_context = retrieve_tables(user_msg)

        if not data_context:
            data_context = "Không tìm thấy bảng nào khớp với câu hỏi. Hãy tự suy luận."

        # 2. XÂY DỰNG PROMPT CHUYÊN SÂU
        system_prompt = f"""Bạn là chuyên gia BigQuery SQL cao cấp.

[CẤU TRÚC DỮ LIỆU ĐƯỢC CUNG CẤP]:
---
[PHẦN 1: LOGIC NGHIỆP VỤ & MAPPING (BẮT BUỘC ĐỌC)]:
{logic_context}
---
[PHẦN 2: BẢNG DỮ LIỆU LIÊN QUAN (TRA CỨU)]:
{data_context}
---

[NHIỆM VỤ]:
Viết câu lệnh SQL Standard trả lời câu hỏi của user: "{user_msg}"

[QUY TẮC QUAN TRỌNG - BẮT BUỘC TUÂN THỦ]:
1. **Logic Mapping (QUAN TRỌNG NHẤT):**
   - Hãy tự đọc phần `[ROUTINE / FUNCTION]` ở trên.
   - Tìm các mệnh đề `CASE WHEN` bên trong code SQL của routine để hiểu ý nghĩa các con số (ID).
   - Ví dụ: Nếu thấy `WHEN status_id = 2 THEN 'New'`, và user hỏi về 'New', bạn PHẢI dùng `status_id = 2`.
   - KHÔNG ĐƯỢC ĐOÁN MÒ. Nếu routine định nghĩa khác, hãy theo routine.

2. **Kỹ thuật BigQuery:**
   - ❌ KHÔNG dùng Correlated Subqueries (Subquery phụ thuộc bảng ngoài).
   - ✅ Dùng JOIN (LEFT JOIN) kết hợp GROUP BY nếu cần.
   - Phải sử dụng các hàm, syntax theo chuẩn cấu trúc của BigQuery.

3. Chỉ trả về code SQL trong ```sql ... ```.

4. Có thể giải thích ngắn gọn sau phần code nếu cần thiết.
"""

        messages_payload = [{"role": "system", "content": system_prompt}]
        
        # Thêm lịch sử (Bộ nhớ ngắn hạn)
        history = get_chat_history_formatted(session_id, limit=8)
        for msg in history:
            if msg['content'] != user_msg: messages_payload.append(msg)
        messages_payload.append({"role": "user", "content": user_msg})

        # Gọi AI
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        response = client.chat(
            model=MODEL_NAME, 
            messages=messages_payload, 
            stream=False, 
            options={"temperature": 0.1} # Nhiệt độ thấp để chính xác
        )
        
        reply = response['message']['content']
        save_message(session_id, "assistant", reply)

        return jsonify({"response": reply})

    except Exception as e:
        print(f"Lỗi: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reload', methods=['POST'])
def reload_schema():
    init_advanced_rag()
    return jsonify({"status": "success", "message": "Reloaded!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
