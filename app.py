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

# --- 1. SETUP MÔI TRƯỜNG ---
# Tắt cảnh báo token (dù không dùng HF nữa nhưng cứ để tránh lỗi env cũ)
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

load_dotenv()
app = Flask(__name__)
CORS(app) # Bật CORS để tránh lỗi kết nối Frontend

# --- 2. CẤU HÌNH ---
SCHEMA_FOLDER = "./schemas"
# Cấu hình Database (Tự động thích ứng SQLite/Postgres cho Render)
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
# Kỹ thuật: Full Context Loading - Nạp 100% dữ liệu vào RAM
GLOBAL_FULL_SCHEMA = ""

# =========================================================
#  PHẦN 3: QUẢN LÝ DATABASE (SQLAlchemy)
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
        print("✅ Database Connected (SQLite/PostgreSQL).")

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
        session = Session.query.get(session_id)
        if not session:
            title = (first_msg[:50] + '...') if len(first_msg) > 50 else first_msg
            db.session.add(Session(id=session_id, title=title))
            db.session.commit()
    except Exception as e:
        print(f"Error creating session: {e}")
        db.session.rollback()

def get_chat_history_formatted(session_id, limit=10):
    try:
        msgs = Message.query.filter_by(session_id=session_id).order_by(desc(Message.created_at)).limit(limit).all()
        history = []
        for m in msgs[::-1]: 
            history.append({"role": m.role, "content": m.content})
        return history
    except:
        return []

# =========================================================
#  PHẦN 4: LOAD TOÀN BỘ SCHEMA (FULL CONTEXT KNOWLEDGE)
# =========================================================
def load_all_schemas():
    """
    Đọc nguyên văn toàn bộ file JSON và nạp vào biến GLOBAL_FULL_SCHEMA.
    AI sẽ đọc trực tiếp từ biến này, đảm bảo không bao giờ lỗi thiếu thư viện hay sót dữ liệu.
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
                    # --- XỬ LÝ TABLE ---
                    if 'table_name' in item:
                        name = item.get('table_name', 'Unknown')
                        desc = item.get('description', '')
                        cols = [f"- {c['name']} ({c.get('type')})" for c in item.get('columns', [])]
                        col_str = "\n".join(cols)
                        
                        schema_parts.append(f"""
[TABLE SCHEMA]
Name: `{name}`
Description: {desc}
Columns:
{col_str}
""")
                    
                    # --- XỬ LÝ ROUTINE (Hàm Logic - Quan trọng nhất) ---
                    elif 'routine_name' in item:
                        name = item.get('routine_name', 'Unknown')
                        # Lấy code SQL gốc để AI tự đọc logic CASE WHEN
                        definition = item.get('routine_definition') or item.get('ddl') or ''
                        desc = item.get('description', '')
                        
                        schema_parts.append(f"""
[ROUTINE / FUNCTION]
Name: `{name}`
Description: {desc}
DEFINITION (SOURCE SQL CODE):
```sql
{definition}
```
(AI NOTE: Hãy đọc kỹ code SQL trên. Nếu có CASE WHEN, dùng nó để map giá trị ID tương ứng)
""")

        except Exception as e:
            print(f"❌ Lỗi đọc file {file_path}: {e}")

    # Gộp tất cả lại thành 1 chuỗi văn bản lớn
    GLOBAL_FULL_SCHEMA = "\n----------------------------------------\n".join(schema_parts)
    print(f"✅ Đã nạp xong! Tổng dung lượng Context: {len(GLOBAL_FULL_SCHEMA)} ký tự.")

# --- KHỞI CHẠY ---
init_db()
load_all_schemas()

# =========================================================
#  PHẦN 5: API ROUTES
# =========================================================

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    try:
        sessions = Session.query.order_by(desc(Session.created_at)).all()
        return jsonify([{'id': s.id, 'title': s.title, 'created_at': s.created_at} for s in sessions])
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
        return jsonify({"status": "success", "message": "Deleted all history."})
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

    if not api_key or not session_id: return jsonify({"error": "Thiếu thông tin"}), 400

    try:
        create_session_if_not_exists(session_id, user_msg)
        save_message(session_id, "user", user_msg)

        # 2. XÂY DỰNG PROMPT CAO CẤP (Chain-of-Thought)
        # Ép AI phải suy luận logic từ Routine trước khi viết Code
        system_prompt = f"""Bạn là chuyên gia BigQuery SQL cao cấp.

[DỮ LIỆU TOÀN CỤC CỦA HỆ THỐNG]:
Dưới đây là TOÀN BỘ Bảng và Hàm (Routine) của hệ thống. Hãy đọc hết để hiểu ngữ cảnh:

{GLOBAL_FULL_SCHEMA}

[YÊU CẦU]:
Viết câu lệnh SQL Standard trả lời câu hỏi: "{user_msg}"

[QUY TRÌNH SUY LUẬN (BẮT BUỘC THỰC HIỆN TRONG ĐẦU)]:
1. **Phân tích câu hỏi:** User đang hỏi về đối tượng nào?
2. **Tra cứu Logic (QUAN TRỌNG NHẤT):**
   - Tìm các `[ROUTINE]` có liên quan đến trạng thái hoặc loại hình (status, type...).
   - Đọc kỹ code SQL bên trong routine (đặc biệt là mệnh đề `CASE WHEN`).
    - Xác định giá trị ID tương ứng với mô tả user hỏi.
   - KHÔNG ĐƯỢC ĐOÁN MÒ.
3. **Viết SQL:**
   - ❌ KHÔNG dùng Correlated Subqueries (Subquery phụ thuộc bảng ngoài).
   - ✅ Dùng JOIN (LEFT JOIN) kết hợp GROUP BY nếu cần.
   - Phải sử dụng các hàm, syntax theo chuẩn cấu trúc của BigQuery.

[OUTPUT]:
4. Chỉ trả về code SQL trong ```sql ... ```.

5. Có thể giải thích ngắn gọn sau phần code nếu cần thiết.
"""

        messages_payload = [{"role": "system", "content": system_prompt}]
        
        # Thêm lịch sử (Bộ nhớ ngắn hạn)
        history = get_chat_history_formatted(session_id, limit=10)
        for msg in history:
            if msg['content'] != user_msg: messages_payload.append(msg)
        messages_payload.append({"role": "user", "content": user_msg})

        # Gọi AI
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        response = client.chat(
            model=MODEL_NAME, 
            messages=messages_payload, 
            stream=False, 
            options={"temperature": 0.05} # Nhiệt độ cực thấp để đảm bảo chính xác logic
        ) 
        
        reply = response['message']['content']
        save_message(session_id, "assistant", reply)

        return jsonify({"response": reply})

    except Exception as e:
        print(f"Lỗi: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reload', methods=['POST'])
def reload_schema():
    load_all_schemas()
    return jsonify({"status": "success", "message": "Reloaded!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
