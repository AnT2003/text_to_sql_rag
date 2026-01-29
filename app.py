import os
import json
import glob
import datetime
import psycopg2
import psycopg2.extras
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client

# --- 1. SETUP MÔI TRƯỜNG ---
load_dotenv()
app = Flask(__name__)

# --- 2. CẤU HÌNH HỆ THỐNG ---
SCHEMA_FOLDER = "./schemas"
# Lấy URL kết nối DB từ biến môi trường (Render sẽ cung cấp biến DATABASE_URL)
# Mặc định fallback về localhost nếu chạy local mà không có env
DATABASE_URL = os.getenv("DATABASE_URL")

# Cấu hình Ollama (Cloud hoặc Local)
OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gpt-oss:120b"
# API Key (Ưu tiên lấy từ .env)
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY") 

# BIẾN TOÀN CỤC: Chứa toàn bộ kiến thức về Database
# Hệ thống sẽ nạp 100% Bảng và Hàm vào đây để AI đọc mỗi lần chat
GLOBAL_FULL_SCHEMA = ""

# =========================================================
#  PHẦN 3: QUẢN LÝ DATABASE (POSTGRESQL) - LƯU LỊCH SỬ CHAT
# =========================================================

def get_db_connection():
    """Tạo kết nối đến PostgreSQL"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"❌ Lỗi kết nối Database: {e}")
        return None

def init_db():
    """Khởi tạo database PostgreSQL nếu chưa có bảng"""
    conn = get_db_connection()
    if not conn: return
    
    try:
        cur = conn.cursor()
        # Tạo bảng sessions
        cur.execute('''CREATE TABLE IF NOT EXISTS sessions 
                     (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP)''')
        
        # Tạo bảng messages (Dùng SERIAL cho id tự tăng trong Postgres)
        cur.execute('''CREATE TABLE IF NOT EXISTS messages 
                     (id SERIAL PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, created_at TIMESTAMP)''')
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Đã khởi tạo Database thành công.")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Database: {e}")

def get_chat_history_formatted(session_id, limit=10):
    """Lấy lịch sử chat của một phiên cụ thể"""
    conn = get_db_connection()
    if not conn: return []
    
    # Sử dụng RealDictCursor để lấy dữ liệu dạng Dictionary
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    # Postgres dùng %s thay vì ? cho tham số
    cur.execute("SELECT role, content FROM messages WHERE session_id = %s ORDER BY created_at DESC LIMIT %s", (session_id, limit))
    rows = cur.fetchall()
    
    conn.close()
    
    history = []
    # Đảo ngược để xếp theo thứ tự thời gian cũ -> mới (User hỏi -> AI trả lời)
    for r in rows[::-1]:
        history.append({"role": r["role"], "content": r["content"]})
    return history

def save_message(session_id, role, content):
    """Lưu tin nhắn vào DB"""
    conn = get_db_connection()
    if not conn: return

    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (%s, %s, %s, %s)", 
                  (session_id, role, content, datetime.datetime.now()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Lỗi lưu tin nhắn: {e}")

def create_session_if_not_exists(session_id, first_msg):
    """Tạo phiên chat mới nếu chưa tồn tại"""
    conn = get_db_connection()
    if not conn: return

    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM sessions WHERE id = %s", (session_id,))
        if not cur.fetchone():
            # Lấy 50 ký tự đầu của tin nhắn làm tiêu đề
            cur.execute("INSERT INTO sessions (id, title, created_at) VALUES (%s, %s, %s)", 
                      (session_id, first_msg[:50], datetime.datetime.now()))
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"Lỗi tạo session: {e}")

def delete_session_data(session_id):
    """Xóa toàn bộ lịch sử của một session"""
    conn = get_db_connection()
    if not conn: return False
    
    try:
        cur = conn.cursor()
        # Xóa tin nhắn trước
        cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
        # Xóa session sau
        cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Lỗi xóa session: {e}")
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
    conn = get_db_connection()
    if not conn: return jsonify([])
    
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM sessions ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    
    # Convert datetime objects to string if needed implies standard JSON serialization handles it usually, 
    # but strictly jsonify handles datetime objects well in newer Flask versions.
    return jsonify([dict(r) for r in rows])

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
