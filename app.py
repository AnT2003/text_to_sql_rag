import os
import json
import glob
import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc

# --- 1. SETUP MÔI TRƯỜNG ---
load_dotenv()
app = Flask(__name__)

# --- 2. CẤU HÌNH ---
SCHEMA_FOLDER = "./schemas"
OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gpt-oss:120b"
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY") 

# Cấu hình Database (Hỗ trợ cả SQLite và PostgreSQL trên Render)
db_url = os.getenv("DATABASE_URL", "sqlite:///chat_history.db")
# Fix lỗi nhỏ của Render (Render dùng postgres://, thư viện mới cần postgresql://)
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# BIẾN TOÀN CỤC CHỨA SCHEMA
GLOBAL_FULL_SCHEMA = ""

# =========================================================
#  PHẦN 3: DATABASE MODELS (SQLAlchemy)
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
    role = db.Column(db.String(20), nullable=False) # user / assistant
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Khởi tạo DB
def init_db():
    with app.app_context():
        db.create_all()
        print("✅ Database đã được khởi tạo/kết nối.")

# =========================================================
#  PHẦN 4: LOGIC DATABASE HELPER
# =========================================================
def create_session_if_not_exists(session_id, first_msg):
    # Kiểm tra xem session có chưa
    session = Session.query.get(session_id)
    if not session:
        # Tạo tiêu đề từ 50 ký tự đầu
        title = (first_msg[:50] + '...') if len(first_msg) > 50 else first_msg
        new_session = Session(id=session_id, title=title)
        db.session.add(new_session)
        db.session.commit()

def save_message(session_id, role, content):
    new_msg = Message(session_id=session_id, role=role, content=content)
    db.session.add(new_msg)
    db.session.commit()

def get_chat_history_formatted(session_id, limit=10):
    # Lấy tin nhắn mới nhất, sau đó đảo ngược lại
    messages = Message.query.filter_by(session_id=session_id)\
        .order_by(desc(Message.created_at))\
        .limit(limit).all()
    
    history = []
    for msg in messages[::-1]: # Đảo ngược thành cũ -> mới
        history.append({"role": msg.role, "content": msg.content})
    return history

# =========================================================
#  PHẦN 5: LOAD SCHEMA (FULL CONTEXT)
# =========================================================
def load_all_schemas():
    global GLOBAL_FULL_SCHEMA
    print("🚀 Đang nạp TOÀN BỘ Schemas vào bộ nhớ...")
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
                    if 'table_name' in item:
                        name = item.get('table_name', 'Unknown')
                        cols = [f"- {c['name']} ({c.get('type')})" for c in item.get('columns', [])]
                        schema_parts.append(f"[TABLE SCHEMA]\nName: `{name}`\nColumns:\n{chr(10).join(cols)}")
                    elif 'routine_name' in item:
                        name = item.get('routine_name', 'Unknown')
                        definition = item.get('routine_definition') or item.get('ddl') or ''
                        schema_parts.append(f"[ROUTINE]\nName: `{name}`\nDEFINITION:\n```sql\n{definition}\n```")
        except Exception as e:
            print(f"❌ Lỗi file {file_path}: {e}")

    GLOBAL_FULL_SCHEMA = "\n----------------------------------------\n".join(schema_parts)
    print(f"✅ Đã nạp xong! Dung lượng: {len(GLOBAL_FULL_SCHEMA)} ký tự.")

# --- KHỞI CHẠY ---
init_db()
load_all_schemas()

# =========================================================
#  PHẦN 6: API ROUTES
# =========================================================

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    # Lấy danh sách session giảm dần theo thời gian
    sessions = Session.query.order_by(desc(Session.created_at)).all()
    return jsonify([{'id': s.id, 'title': s.title, 'created_at': s.created_at} for s in sessions])

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id): 
    # Lấy toàn bộ lịch sử (limit 100) để hiển thị UI
    msgs = Message.query.filter_by(session_id=session_id).order_by(Message.created_at).limit(100).all()
    return jsonify([{'role': m.role, 'content': m.content} for m in msgs])

# --- API MỚI: XÓA LỊCH SỬ ---
@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    try:
        # Xóa toàn bộ dữ liệu bảng messages và sessions
        Message.query.delete()
        Session.query.delete()
        db.session.commit()
        return jsonify({"status": "success", "message": "Đã xóa toàn bộ lịch sử chat!"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    global GLOBAL_FULL_SCHEMA
    data = request.json
    api_key = data.get('api_key') or DEFAULT_API_KEY
    user_msg = data.get('message')
    session_id = data.get('session_id')

    if not api_key or not session_id: return jsonify({"error": "Thiếu thông tin"}), 400

    try:
        create_session_if_not_exists(session_id, user_msg)
        save_message(session_id, "user", user_msg)

        # Prompt Full Context
        system_prompt = f"""Bạn là một chuyên gia BigQuery SQL cao cấp.

[DỮ LIỆU CỦA HỆ THỐNG]:
Dưới đây là toàn bộ Bảng và Hàm (Routine). Bạn có quyền truy cập tất cả:

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

3. Chỉ trả về code SQL trong ```sql ... ```.

4. Có thể giải thích ngắn gọn sau phần code nếu cần thiết.
"""
        messages_payload = [{"role": "system", "content": system_prompt}]
        
        # Thêm lịch sử chat gần nhất (bộ nhớ ngắn hạn)
        history = get_chat_history_formatted(session_id, limit=6)
        for msg in history:
            if msg['content'] != user_msg: 
                messages_payload.append(msg)
        
        messages_payload.append({"role": "user", "content": user_msg})

        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        response = client.chat(model=MODEL_NAME, messages=messages_payload, stream=False, options={"temperature": 0.1}) 
        
        ai_reply = response['message']['content']
        save_message(session_id, "assistant", ai_reply)

        return jsonify({"response": ai_reply})

    except Exception as e:
        print(f"Lỗi: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reload', methods=['POST'])
def reload_schema():
    load_all_schemas()
    return jsonify({"status": "success", "message": "Reloaded!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
