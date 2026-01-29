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

# --- 1. SETUP MÔI TRƯỜNG ---
# Tắt cảnh báo token của HuggingFace
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# --- 2. IMPORT THƯ VIỆN RAG (Giữ lại import để tránh lỗi nếu mở rộng sau này) ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

load_dotenv()
app = Flask(__name__)
CORS(app) # Bật CORS để tránh lỗi kết nối trên Render

# --- 3. CẤU HÌNH ---
SCHEMA_FOLDER = "./schemas"

# CẤU HÌNH DATABASE (QUAN TRỌNG: Tự động chọn SQLite hoặc Postgres)
# Nếu chạy local: dùng sqlite:///chat_history.db
# Nếu chạy Render: dùng biến môi trường DATABASE_URL
db_url = os.getenv("DATABASE_URL", "sqlite:///chat_history.db")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gpt-oss:120b"
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY") 

# BIẾN TOÀN CỤC: Chứa toàn bộ kiến thức về Database
GLOBAL_FULL_SCHEMA = ""

# =========================================================
#  PHẦN 3: QUẢN LÝ DATABASE (SQLAlchemy - Postgres Compatible)
# =========================================================

# 1. Định nghĩa bảng Sessions
class Session(db.Model):
    __tablename__ = 'sessions'
    id = db.Column(db.String(50), primary_key=True)
    title = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# 2. Định nghĩa bảng Messages
class Message(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), db.ForeignKey('sessions.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

def init_db():
    """Khởi tạo database (Tự tạo bảng nếu chưa có)"""
    with app.app_context():
        db.create_all()
        print("✅ Database đã được khởi tạo thành công (SQLite/Postgres).")

def get_chat_history_formatted(session_id, limit=10):
    """Lấy lịch sử chat format chuẩn cho AI"""
    # Lấy tin nhắn mới nhất theo thời gian giảm dần
    msgs = Message.query.filter_by(session_id=session_id).order_by(desc(Message.created_at)).limit(limit).all()
    
    history = []
    # Đảo ngược để xếp theo thứ tự thời gian cũ -> mới (để AI hiểu ngữ cảnh)
    for msg in msgs[::-1]:
        history.append({"role": msg.role, "content": msg.content})
    return history

def save_message(session_id, role, content):
    """Lưu tin nhắn vào DB"""
    try:
        new_msg = Message(session_id=session_id, role=role, content=content)
        db.session.add(new_msg)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Lỗi lưu message: {e}")

def create_session_if_not_exists(session_id, first_msg):
    """Tạo phiên chat mới nếu chưa tồn tại"""
    try:
        session = Session.query.get(session_id)
        if not session:
            # Lấy 50 ký tự đầu làm tiêu đề
            title = (first_msg[:50] + '...') if len(first_msg) > 50 else first_msg
            new_session = Session(id=session_id, title=title)
            db.session.add(new_session)
            db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Lỗi tạo session: {e}")

# =========================================================
#  PHẦN 4: KỸ THUẬT FULL-CONTEXT LOADING (ĐỌC TOÀN BỘ)
# =========================================================
def load_all_schemas():
    """
    Kỹ thuật Advanced: Đọc TẤT CẢ file trong thư mục schemas và gộp lại nguyên bản.
    Không dùng Regex cắt gọt, để AI tự đọc Raw Data (DDL/Definition) để hiểu ngữ cảnh sâu nhất.
    """
    global GLOBAL_FULL_SCHEMA
    print("🚀 Đang nạp TOÀN BỘ Schemas vào bộ nhớ (Full Context)...")
    
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
                    # --- XỬ LÝ TABLE (BẢNG) ---
                    if 'table_name' in item:
                        name = item.get('table_name', 'Unknown')
                        desc = item.get('description', '')
                        # Format đơn giản: Tên cột (Kiểu dữ liệu)
                        cols = [f"- {c['name']} ({c.get('type')})" for c in item.get('columns', [])]
                        col_str = "\n".join(cols)
                        
                        schema_parts.append(f"""
[TABLE SCHEMA]
Name: `{name}`
Description: {desc}
Columns:
{col_str}
""")
                    
                    # --- XỬ LÝ ROUTINE (HÀM - QUAN TRỌNG NHẤT) ---
                    elif 'routine_name' in item:
                        name = item.get('routine_name', 'Unknown')
                        # Lấy code SQL gốc (quan trọng nhất để hiểu logic CASE WHEN)
                        # Ưu tiên ddl, nếu không có thì lấy routine_definition
                        definition = item.get('ddl') or item.get('routine_definition') or ''
                        desc = item.get('description', '')
                        
                        schema_parts.append(f"""
[ROUTINE / FUNCTION]
Name: `{name}`
Description: {desc}
DEFINITION (SOURCE SQL CODE):
```sql
{definition}
```
(AI NOTE: Hãy đọc kỹ code SQL trên. Nếu có CASE WHEN, hãy dùng nó để map giá trị ID tương ứng)
""")

        except Exception as e:
            print(f"❌ Lỗi đọc file {file_path}: {e}")

    # Gộp tất cả lại thành 1 chuỗi văn bản lớn
    GLOBAL_FULL_SCHEMA = "\n----------------------------------------\n".join(schema_parts)
    print(f"✅ Đã nạp xong! Tổng dung lượng Context: {len(GLOBAL_FULL_SCHEMA)} ký tự.")

# --- KHỞI CHẠY LẦN ĐẦU ---
# Đảm bảo chạy khi file được import hoặc thực thi
init_db()
load_all_schemas()

# =========================================================
#  PHẦN 5: API ROUTES & LOGIC CHAT
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """API lấy danh sách các phiên chat"""
    try:
        # Sử dụng SQLAlchemy để query, sắp xếp mới nhất lên đầu
        sessions = Session.query.order_by(desc(Session.created_at)).all()
        # Chuyển đổi object thành dict
        return jsonify([{'id': s.id, 'title': s.title, 'created_at': s.created_at} for s in sessions])
    except Exception as e:
        print(f"Lỗi lấy session: {e}")
        return jsonify([])

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id): 
    """API lấy nội dung chat"""
    return jsonify(get_chat_history_formatted(session_id, limit=50))

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """API Xóa toàn bộ lịch sử (Dùng khi cần reset)"""
    try:
        # Xóa hết dữ liệu
        Message.query.delete()
        Session.query.delete()
        db.session.commit()
        return jsonify({"status": "success", "message": "Đã xóa toàn bộ lịch sử chat!"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    # Sử dụng biến toàn cục chứa toàn bộ schema
    global GLOBAL_FULL_SCHEMA
    
    data = request.json
    api_key = data.get('api_key') or DEFAULT_API_KEY
    user_msg = data.get('message')
    session_id = data.get('session_id')

    if not api_key: return jsonify({"error": "Thiếu API Key"}), 401
    if not session_id: return jsonify({"error": "Thiếu Session ID"}), 400

    try:
        # 1. Lưu Session và Tin nhắn User
        create_session_if_not_exists(session_id, user_msg)
        save_message(session_id, "user", user_msg)

        # 2. XÂY DỰNG PROMPT CAO CẤP (Đưa toàn bộ Schema vào)
        # Đây là kỹ thuật "In-Context Learning": Dạy AI bằng chính dữ liệu của bạn ngay trong prompt.
        system_prompt = f"""Bạn là một chuyên gia BigQuery SQL cao cấp.

[DỮ LIỆU CỦA HỆ THỐNG]:
Dưới đây là toàn bộ Bảng và Hàm (Routine) bạn có quyền truy cập. 
HÃY ĐỌC KỸ TOÀN BỘ ĐỂ HIỂU LOGIC DỮ LIỆU:

{GLOBAL_FULL_SCHEMA}

[YÊU CẦU]:
Viết câu lệnh SQL Standard trả lời câu hỏi của user.

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
   - Tên bảng phải đặt trong dấu backtick (`).

3. Chỉ trả về code SQL trong ```sql ... ```.

4. Có thể giải thích ngắn gọn sau phần code nếu cần thiết.
"""

        messages_payload = [{"role": "system", "content": system_prompt}]
        
        # Thêm lịch sử chat gần nhất để AI nhớ ngữ cảnh
        history = get_chat_history_formatted(session_id, limit=10)
        for msg in history:
            if msg['content'] != user_msg: 
                messages_payload.append(msg)
        
        # Thêm câu hỏi hiện tại
        messages_payload.append({"role": "user", "content": user_msg})

        # 3. Gọi AI
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        
        # Temperature = 0.1: Giữ cho AI đủ sáng tạo để viết SQL nhưng vẫn tuân thủ dữ liệu
        response = client.chat(
            model=MODEL_NAME, 
            messages=messages_payload, 
            stream=False, 
            options={"temperature": 0.1}
        )
        
        ai_reply = response['message']['content']
        
        # 4. Lưu câu trả lời của AI
        save_message(session_id, "assistant", ai_reply)

        return jsonify({"response": ai_reply})

    except Exception as e:
        print(f"Lỗi Server: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reload', methods=['POST'])
def reload_schema():
    """API để nạp lại dữ liệu khi bạn sửa file JSON"""
    load_all_schemas()
    return jsonify({"status": "success", "message": "Đã nạp lại toàn bộ dữ liệu Schema!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
