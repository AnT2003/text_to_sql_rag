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

# Cấu hình AI Ollama
OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gemini-3-flash-preview:latest"  # Thay đổi model tùy vào setup thực tế
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY")
SCHEMA_FOLDER = "./schemas"

# BIẾN TOÀN CỤC: Chứa toàn bộ kiến thức về Database
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
            # Tạo title ngắn gọn từ tin nhắn đầu tiên
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
        # Đảo ngược lại để đúng thứ tự thời gian khi đưa vào Prompt
        for m in msgs[::-1]:
            history.append({"role": m.role, "content": m.content})
        return history
    except:
        return []

# =========================================================
#  PHẦN 4: LOAD SCHEMA (STRICT MODE - LOGIC QUAN TRỌNG)
# =========================================================

def load_all_schemas():
    """
    Hàm này đọc file JSON schema và tạo ra Context cực kỳ chi tiết.
    Nó lấy cả Dataset ID để đảm bảo query đúng bảng BigQuery.
    """
    global GLOBAL_FULL_SCHEMA
    print("🚀 Đang nạp TOÀN BỘ Schemas vào bộ nhớ (Strict Context Mode)...")

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
                    # --- 1. Xác định Dataset ID & Project ID ---
                    table_ref = item.get('tableReference', {})
                    dataset_id = table_ref.get('datasetId') or item.get('dataset_id', 'UnknownDataset')
                    project_id = table_ref.get('projectId') or item.get('project_id', '')

                    # Prefix đầy đủ: `project.dataset` hoặc `dataset`
                    full_prefix = f"{project_id}.{dataset_id}" if project_id else dataset_id

                    # --- 2. XỬ LÝ TABLE (BẢNG DỮ LIỆU) ---
                    if 'table_name' in item or 'tableId' in table_ref:
                        table_name = item.get('table_name') or table_ref.get('tableId')
                        description = item.get('description', 'No description.')

                        # Xử lý danh sách cột chi tiết
                        cols = []
                        for c in item.get('columns', []):
                            c_name = c.get('name')
                            c_type = c.get('type')
                            c_desc = c.get('description', '')
                            col_str = f"- `{c_name}` ({c_type})"
                            if c_desc: col_str += f": {c_desc}"
                            cols.append(col_str)

                        columns_block = "\n".join(cols)

                        schema_parts.append(f"""
[TABLE ENTITY]
Full Name: `{full_prefix}.{table_name}`
Description: {description}
COLUMNS DEFINITION (ONLY USE THESE COLUMNS):
{columns_block}
""")

                    # --- 3. XỬ LÝ ROUTINE / FUNCTION (LOGIC NGHIỆP VỤ) ---
                    elif 'routine_name' in item or 'routineId' in item.get('routineReference', {}):
                        routine_ref = item.get('routineReference', {})
                        routine_name = item.get('routine_name') or routine_ref.get('routineId')

                        # Lấy code SQL gốc
                        definition = item.get('routine_definition') or item.get('ddl') or ''
                        description = item.get('description', '')

                        schema_parts.append(f"""
[LOGIC ROUTINE / FUNCTION]
Full Name: `{full_prefix}.{routine_name}`
Description: {description}
SOURCE CODE (READ CAREFULLY TO MAP VALUES/STATUS):
```sql
{definition}
```
""")

        except Exception as e:
            print(f"❌ Lỗi đọc file {file_path}: {e}")

    # Gộp tất cả thành 1 biến String khổng lồ
    GLOBAL_FULL_SCHEMA = "\n----------------------------------------\n".join(schema_parts)
    print(f"✅ Đã nạp xong! Tổng dung lượng Context: {len(GLOBAL_FULL_SCHEMA)} ký tự.")

# --- Gọi hàm khởi tạo ---
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
        return jsonify([{'id': s.id, 'title': s.title, 'created_at': s.created_at} for s in sessions])
    except:
        return jsonify([])

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
    # Sử dụng biến Global chứa Schema đã load
    global GLOBAL_FULL_SCHEMA

    data = request.json
    api_key = data.get('api_key') or DEFAULT_API_KEY
    user_msg = data.get('message')
    session_id = data.get('session_id')

    if not api_key or not session_id:
        return jsonify({"error": "Thiếu thông tin API Key hoặc Session ID"}), 400

    try:
        # 1. Lưu tin nhắn User
        create_session_if_not_exists(session_id, user_msg)
        save_message(session_id, "user", user_msg)

        # 2. XÂY DỰNG PROMPT (ANTI-HALLUCINATION)
        system_prompt = f"""Bạn là chuyên gia SQL BigQuery.
Nhiệm vụ: Chuyển câu hỏi người dùng thành câu lệnh SQL Standard.

[NGUYÊN TẮC BẮT BUỘC - KHÔNG ĐƯỢC VI PHẠM]:
1. **Nguồn sự thật duy nhất:** Chỉ được sử dụng các bảng và cột được liệt kê trong phần [DATABASE SCHEMA] bên dưới. KHÔNG ĐƯỢC TỰ BỊA TÊN CỘT (như created_at, id, name) nếu schema không có.
2. **Định danh đầy đủ:** Luôn sử dụng tên bảng dạng `dataset.table` (Full Qualified Name) và lấy đúng như tên bảng trong schema table trong [DATABASE SCHEMA], không được tự ý bịa ra hoặc giả định thêm.
3. **Mapping Logic:**
   - Nếu User yêu cầu truy vấn có điều kiện kèm theo, bạn PHẢI tham khảo thêm phần [LOGIC ROUTINE] để hiểu rõ ý nghĩa các trường dữ liệu, không được tự suy diễn..
   - Tìm trong code SQL của routine (mệnh đề `CASE WHEN`) để xem trạng thái đó ứng với số ID nào.
   - Ví dụ: Thấy `WHEN status=4 THEN 'Approved'` thì phải query `WHERE status = 4`.
4. **Kỹ thuật BigQuery:**
   - ❌ KHÔNG dùng Correlated Subqueries (Subquery phụ thuộc bảng ngoài).
   - ✅ Dùng JOIN (LEFT JOIN) kết hợp GROUP BY nếu cần.
   - Phải sử dụng các hàm, syntax theo chuẩn cấu trúc của BigQuery.

[ĐỊNH DẠNG TRẢ VỀ]:

1. Chỉ trả về code SQL trong ```sql ... ```.

2. Có thể giải thích ngắn gọn sau phần code nếu cần thiết.
"""

        messages_payload = [{"role": "system", "content": system_prompt}]

        # Thêm context hội thoại gần nhất
        history = get_chat_history_formatted(session_id, limit=8)
        for msg in history:
            if msg['content'] != user_msg:
                messages_payload.append(msg)

        messages_payload.append({"role": "user", "content": user_msg})

        # 3. GỌI OLLAMA API
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})

        response = client.chat(
            model=MODEL_NAME,
            messages=messages_payload,
            stream=False,
            # QUAN TRỌNG: temperature=0.0 để loại bỏ tính ngẫu nhiên, ép AI chỉ dựa vào dữ liệu có thật
            options={"temperature": 0.0, "top_p": 0.1}
        )

        reply = response['message']['content']

        # 4. Lưu câu trả lời Assistant
        save_message(session_id, "assistant", reply)

        return jsonify({"response": reply})

    except Exception as e:
        print(f"Lỗi xử lý Chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reload', methods=['POST'])
def reload_schema():
    load_all_schemas()
    return jsonify({"status": "success", "message": "Schemas reloaded successfully!"})

# =========================================================
#  PHẦN 6: MAIN ENTRY
# =========================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
