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

# =========================================================
#  PHẦN 1: SETUP MÔI TRƯỜNG & CẤU HÌNH
# =========================================================

# Tắt cảnh báo token huggingface không cần thiết
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# --- 1. SETUP MÔI TRƯỜNG ---
load_dotenv()
app = Flask(__name__)
CORS(app)  # Bật CORS cho Frontend

# Cấu hình Database (Tự động thích ứng SQLite/Postgres cho Render/Local)
db_url = os.getenv("DATABASE_URL", "sqlite:///chat_history.db")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- 2. CẤU HÌNH HỆ THỐNG ---
SCHEMA_FOLDER = "./schemas"

# Cấu hình AI Ollama
# Cấu hình Ollama (Cloud hoặc Local)
OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gpt-oss:120b"
# API Key (Ưu tiên lấy từ .env)
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY") 

# BIẾN TOÀN CỤC: Chứa toàn bộ kiến thức về Database
# Hệ thống sẽ nạp 100% Bảng và Hàm vào đây để AI đọc mỗi lần chat
GLOBAL_FULL_SCHEMA = ""

# =========================================================
#  PHẦN 2: DATABASE MODELS (SQLAlchemy)
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

# =========================================================
#  PHẦN 3: HÀM HỖ TRỢ DATABASE
# =========================================================

def init_db():
    with app.app_context():
        db.create_all()
        print("✅ Database Connected (SQLite/PostgreSQL).")

def save_message(session_id, role, content):
    """Lưu tin nhắn vào DB"""
    try:
        new_msg = Message(session_id=session_id, role=role, content=content)
        db.session.add(new_msg)
        db.session.commit()
    except Exception as e:
        print(f"Error saving message: {e}")
        db.session.rollback()

def create_session_if_not_exists(session_id, first_msg):
    """Tạo phiên chat mới nếu chưa tồn tại"""
    try:
        session = Session.query.get(session_id)
        if not session:
            # Tạo title ngắn gọn từ tin nhắn đầu tiên
            title = (first_msg[:50] + '...') if len(first_msg) > 50 else first_msg
            db.session.add(Session(id=session_id, title=title))
            db.session.commit()
    except Exception as e:
        print(f"Error creating session: {e}")
        db.session.rollback()

def get_chat_history_formatted(session_id, limit=10):
    """Lấy lịch sử chat của một phiên cụ thể"""
    try:
        msgs = Message.query.filter_by(session_id=session_id).order_by(desc(Message.created_at)).limit(limit).all()
        history = []
        # Đảo ngược lại để đúng thứ tự thời gian khi đưa vào Prompt
        for m in msgs[::-1]:
            history.append({"role": m.role, "content": m.content})
        return history
    except:
        return []

def delete_session_data(session_id):
    """Xóa toàn bộ lịch sử của một session"""
    try:
        Message.query.filter_by(session_id=session_id).delete()
        Session.query.filter_by(id=session_id).delete()
        db.session.commit()
        return True
    except Exception as e:
        print(f"Lỗi xóa session: {e}")
        db.session.rollback()
        return False

# =========================================================
#  PHẦN 4: KỸ THUẬT FULL-CONTEXT LOADING (ĐỌC TOÀN BỘ)
# =========================================================
def load_all_schemas():
    """
    Kỹ thuật Advanced: Đọc TẤT CẢ file trong thư mục schemas và gộp lại nguyên bản.
    Đã sửa: Chỉ lấy name và ddl cho Table; name, definition, ddl, arguments cho Routine.
    Loại bỏ description và không parse columns.
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
                    # Chỉ lấy name và ddl
                    if 'table_name' in item:
                        name = item.get('table_name', 'Unknown')
                        ddl = item.get('ddl', '')
                        
                        schema_parts.append(f"""
[TABLE SCHEMA]
Name: `{name}`
DDL:
```sql
{ddl}
```
""")
                    
                    # --- XỬ LÝ ROUTINE (HÀM) ---
                    # Chỉ lấy routine_name, routine_definition, ddl và arguments
                    elif 'routine_name' in item:
                        name = item.get('routine_name', 'Unknown')
                        ddl = item.get('ddl', '')
                        definition = item.get('routine_definition', '')
                        arguments = item.get('arguments', [])
                        
                        # Format arguments thành chuỗi JSON để dễ đọc
                        if isinstance(arguments, (list, dict)):
                            args_str = json.dumps(arguments, ensure_ascii=False)
                        else:
                            args_str = str(arguments)
                        
                        # Ưu tiên lấy DDL, nếu không có thì lấy routine_definition
                        code_content = ddl if ddl else definition
                        
                        schema_parts.append(f"""
[ROUTINE / FUNCTION]
Name: `{name}`
Arguments: {args_str}
DEFINITION (SOURCE SQL CODE):
```sql
{code_content}
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
        sessions = Session.query.order_by(desc(Session.created_at)).all()
        return jsonify([{'id': s.id, 'title': s.title, 'created_at': s.created_at} for s in sessions])
    except:
        return jsonify([])

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session_endpoint(session_id):
    """API xóa lịch sử chat của một session"""
    success = delete_session_data(session_id)
    if success:
        return jsonify({"status": "success", "message": "Đã xóa lịch sử chat thành công."})
    else:
        return jsonify({"error": "Lỗi khi xóa session"}), 500

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id): 
    """API lấy nội dung chat"""
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

    if not api_key: return jsonify({"error": "Thiếu API Key"}), 401
    if not session_id: return jsonify({"error": "Thiếu Session ID"}), 400

    try:
        # 1. Lưu Session và Tin nhắn User
        create_session_if_not_exists(session_id, user_msg)
        save_message(session_id, "user", user_msg)

        # 2. XÂY DỰNG PROMPT CAO CẤP (Đưa toàn bộ Schema vào)
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
   - KHÔNG ĐƯỢC ĐOÁN MÒ. Nếu routine định nghĩa khác, hãy theo routine.

2. **Kỹ thuật BigQuery:**
   - ❌ KHÔNG dùng Correlated Subqueries (Subquery phụ thuộc bảng ngoài).
   - ✅ Dùng JOIN (LEFT JOIN) kết hợp GROUP BY nếu cần.
   - Phải sử dụng các hàm, syntax theo chuẩn cấu trúc của BigQuery.

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
