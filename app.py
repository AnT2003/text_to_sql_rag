import os
import json
import glob
import sqlite3
import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client

# --- 1. SETUP MÔI TRƯỜNG ---
load_dotenv()
app = Flask(__name__)

# --- 2. CẤU HÌNH HỆ THỐNG ---
SCHEMA_FOLDER = "./schemas"
DB_FILE = "chat_history.db"
# Cấu hình Ollama (Cloud hoặc Local)
OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gpt-oss:120b"
# API Key (Ưu tiên lấy từ .env)
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY") 

# BIẾN TOÀN CỤC: Chứa toàn bộ kiến thức về Database
# Hệ thống sẽ nạp 100% Bảng và Hàm vào đây để AI đọc mỗi lần chat
GLOBAL_FULL_SCHEMA = ""

# =========================================================
#  PHẦN 3: QUẢN LÝ DATABASE (SQLITE) - LƯU LỊCH SỬ CHAT
# =========================================================
def init_db():
    """Khởi tạo database SQLite nếu chưa có"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions 
                 (id TEXT PRIMARY KEY, title TEXT, created_at DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at DATETIME)''')
    conn.commit()
    conn.close()

def get_chat_history_formatted(session_id, limit=10):
    """Lấy lịch sử chat của một phiên cụ thể"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?", (session_id, limit))
    rows = c.fetchall()
    conn.close()
    
    history = []
    # Đảo ngược để xếp theo thứ tự thời gian cũ -> mới (User hỏi -> AI trả lời)
    for r in rows[::-1]:
        history.append({"role": r["role"], "content": r["content"]})
    return history

def save_message(session_id, role, content):
    """Lưu tin nhắn vào DB"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", 
              (session_id, role, content, datetime.datetime.now()))
    conn.commit()
    conn.close()

def create_session_if_not_exists(session_id, first_msg):
    """Tạo phiên chat mới nếu chưa tồn tại"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if not c.fetchone():
        # Lấy 50 ký tự đầu của tin nhắn làm tiêu đề
        c.execute("INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)", 
                  (session_id, first_msg[:50], datetime.datetime.now()))
        conn.commit()
    conn.close()

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
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    rows = c.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id): 
    """API lấy nội dung chat"""
    return jsonify(get_chat_history_formatted(session_id, limit=50))

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
