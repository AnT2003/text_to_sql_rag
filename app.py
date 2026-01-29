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
# Cấu hình Ollama
OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gpt-oss:120b"
# API Key
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY") 

# BIẾN TOÀN CỤC CHỨA DỮ LIỆU
# 1. Store: Chứa full nội dung (DDL, Logic) để lấy ra khi cần (Map: Name -> Content)
GLOBAL_SCHEMA_STORE = {} 
# 2. Index: Chứa danh sách TÊN + Tóm tắt nhẹ để AI quét nhanh (String)
GLOBAL_SCHEMA_INDEX = ""
# 3. List Names: Danh sách tên để đối chiếu
GLOBAL_ALL_NAMES = []

# =========================================================
#  PHẦN 3: QUẢN LÝ DATABASE (SQLITE)
# =========================================================
def init_db():
    """Khởi tạo database và bảng nếu chưa tồn tại"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS sessions 
                     (id TEXT PRIMARY KEY, title TEXT, created_at DATETIME)''')
        c.execute('''CREATE TABLE IF NOT EXISTS messages 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at DATETIME)''')
        conn.commit()
        conn.close()
        print("✅ Database initialized (Sessions & Messages tables ready).")
    except Exception as e:
        print(f"❌ Database init error: {e}")

def get_chat_history_formatted(session_id, limit=10):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?", (session_id, limit))
    rows = c.fetchall()
    conn.close()
    history = []
    for r in rows[::-1]:
        history.append({"role": r["role"], "content": r["content"]})
    return history

def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", 
              (session_id, role, content, datetime.datetime.now()))
    conn.commit()
    conn.close()

def create_session_if_not_exists(session_id, first_msg):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if not c.fetchone():
        c.execute("INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)", 
                  (session_id, first_msg[:50], datetime.datetime.now()))
        conn.commit()
    conn.close()

# =========================================================
#  PHẦN 4: KỸ THUẬT RAG 2 BƯỚC (SMART LOADING)
# =========================================================
def load_all_schemas():
    """
    Nạp dữ liệu theo 2 tầng:
    1. Tầng Index (Nhẹ): Để AI quét chọn lọc.
    2. Tầng Store (Nặng): Chứa nội dung chi tiết.
    """
    global GLOBAL_SCHEMA_STORE, GLOBAL_SCHEMA_INDEX, GLOBAL_ALL_NAMES
    GLOBAL_SCHEMA_STORE = {}
    GLOBAL_ALL_NAMES = []
    index_lines = []
    
    print("🚀 Đang nạp Schemas (Two-Stage RAG Mode)...")
    
    if not os.path.exists(SCHEMA_FOLDER): 
        print(f"⚠️ Không tìm thấy thư mục {SCHEMA_FOLDER}")
        return

    json_files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    name = ""
                    full_content = ""
                    summary = ""

                    # --- TABLE ---
                    if 'table_name' in item:
                        name = item.get('table_name', 'Unknown')
                        ddl = item.get('ddl', '')
                        
                        # Full content (cho Bước 2)
                        full_content = f"[TABLE] Name: `{name}`\nDDL:\n```sql\n{ddl}\n```"
                        
                        # Summary (cho Bước 1 - chỉ cần tên bảng để tiết kiệm token)
                        summary = f"- TABLE: {name}"

                    # --- ROUTINE ---
                    elif 'routine_name' in item:
                        name = item.get('routine_name', 'Unknown')
                        routine_def = item.get('routine_definition', '')
                        ddl = item.get('ddl', '')
                        arguments = item.get('arguments', [])
                        
                        # Full content (cho Bước 2 - Đầy đủ logic)
                        args_str = json.dumps(arguments, ensure_ascii=False)
                        full_content = f"""
[ROUTINE] Name: `{name}`
Arguments: {args_str}
Definition:
```sql
{routine_def}
```
DDL:
```sql
{ddl}
```
"""
                        # Summary (cho Bước 1)
                        summary = f"- ROUTINE: {name}"

                    if name and full_content:
                        GLOBAL_SCHEMA_STORE[name] = full_content
                        GLOBAL_ALL_NAMES.append(name)
                        index_lines.append(summary)

        except Exception as e:
            print(f"❌ Lỗi đọc file {file_path}: {e}")

    # Tạo Index String
    GLOBAL_SCHEMA_INDEX = "\n".join(index_lines)
    print(f"✅ Đã nạp {len(GLOBAL_ALL_NAMES)} đối tượng vào Index.")

# =========================================================
#  PHẦN 5: API ROUTES & LOGIC CHAT (QUAN TRỌNG)
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    rows = c.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id): 
    return jsonify(get_chat_history_formatted(session_id, limit=50))

def ai_select_relevant_schemas(client, user_msg):
    """
    BƯỚC 1: Gửi danh sách toàn bộ tên bảng/hàm cho AI.
    Yêu cầu AI chọn ra những cái tên liên quan nhất.
    """
    if not GLOBAL_SCHEMA_INDEX:
        return []

    # Prompt đặc biệt để chọn lọc
    selection_prompt = f"""Bạn là trợ lý dữ liệu thông minh.
Nhiệm vụ: Dựa vào câu hỏi của người dùng, hãy xác định những Table hoặc Routine nào cần thiết để trả lời.

DANH SÁCH TOÀN BỘ TABLE VÀ ROUTINE HIỆN CÓ:
{GLOBAL_SCHEMA_INDEX}

CÂU HỎI NGƯỜI DÙNG: "{user_msg}"

YÊU CẦU TRẢ VỀ:
- Chỉ liệt kê tên chính xác của các bảng/hàm liên quan.
- Không giải thích gì thêm.
- Nếu cần thiết, hãy chọn dư còn hơn bỏ sót.
"""
    
    try:
        response = client.chat(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": selection_prompt}], 
            stream=False,
            options={"temperature": 0.0} # Temp thấp để chính xác
        )
        ai_response_text = response['message']['content']
        
        # Logic phân tích phản hồi của AI để lấy ra list tên
        # Cách đơn giản và hiệu quả nhất: Quét xem tên nào trong Database có xuất hiện trong câu trả lời của AI
        selected_names = []
        for name in GLOBAL_ALL_NAMES:
            if name in ai_response_text:
                selected_names.append(name)
        
        print(f"🔍 AI đã chọn {len(selected_names)} schemas liên quan: {selected_names}")
        return selected_names
        
    except Exception as e:
        print(f"⚠️ Lỗi bước chọn lọc: {e}")
        return []

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    api_key = data.get('api_key') or DEFAULT_API_KEY
    user_msg = data.get('message')
    session_id = data.get('session_id')

    if not api_key: return jsonify({"error": "Thiếu API Key"}), 401
    if not session_id: return jsonify({"error": "Thiếu Session ID"}), 400

    try:
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        
        # 1. Lưu tin nhắn User
        create_session_if_not_exists(session_id, user_msg)
        save_message(session_id, "user", user_msg)

        # -------------------------------------------------------------
        # BƯỚC 1: AI QUÉT TOÀN BỘ INDEX ĐỂ CHỌN SCHEMA (RAG STAGE 1)
        # -------------------------------------------------------------
        # Thay vì search keyword, ta hỏi thẳng AI
        selected_schema_names = ai_select_relevant_schemas(client, user_msg)
        
        # Fallback: Nếu AI không chọn được gì (hoặc lỗi), ta dùng cơ chế keyword search "thô" để vớt vát
        if not selected_schema_names:
            print("⚠️ AI không chọn được bảng nào, chuyển sang chế độ dự phòng (keyword match)...")
            query_tokens = user_msg.lower().split()
            for name in GLOBAL_ALL_NAMES:
                if any(token in name.lower() for token in query_tokens):
                    selected_schema_names.append(name)
        
        # -------------------------------------------------------------
        # BƯỚC 2: LOAD FULL CONTEXT CHO NHỮNG MỤC ĐÃ CHỌN (RAG STAGE 2)
        # -------------------------------------------------------------
        context_parts = []
        current_chars = 0
        MAX_CHARS = 100000 # Giới hạn an toàn cho bước tạo code
        
        # Luôn ưu tiên những bảng AI đã chọn
        unique_names = list(set(selected_schema_names))
        
        for name in unique_names:
            content = GLOBAL_SCHEMA_STORE.get(name, "")
            if len(context_parts) == 0 or (current_chars + len(content) < MAX_CHARS):
                context_parts.append(content)
                current_chars += len(content)
        
        final_context = "\n--------------------\n".join(context_parts)

        # -------------------------------------------------------------
        # BƯỚC 3: TẠO SQL VỚI FULL CONTEXT ĐÃ CHỌN LỌC
        # -------------------------------------------------------------
        system_prompt = f"""Bạn là chuyên gia BigQuery SQL.

[NGỮ CẢNH DỮ LIỆU ĐÃ ĐƯỢC CHỌN LỌC KỸ]:
Dưới đây là DDL và Logic chi tiết của các Bảng/Routine liên quan trực tiếp đến câu hỏi.
(Đã được lọc từ toàn bộ Database để đảm bảo độ chính xác cao nhất)

{final_context}

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
        
        # Thêm lịch sử chat
        history = get_chat_history_formatted(session_id, limit=5)
        for msg in history:
            if msg['content'] != user_msg:
                messages_payload.append(msg)
        
        messages_payload.append({"role": "user", "content": user_msg})

        # Gọi AI để viết Code
        response = client.chat(
            model=MODEL_NAME, 
            messages=messages_payload, 
            stream=False, 
            options={"temperature": 0.1}
        )
        
        ai_reply = response['message']['content']
        save_message(session_id, "assistant", ai_reply)

        return jsonify({"response": ai_reply})

    except Exception as e:
        print(f"Lỗi Server: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reload', methods=['POST'])
def reload_schema():
    load_all_schemas()
    return jsonify({"status": "success", "message": "Đã nạp lại dữ liệu (Mode: Two-Stage RAG)!"})

# --- KHỞI CHẠY HỆ THỐNG ---
# Chạy ngay khi import để tránh lỗi 'No such table' khi dùng flask run
init_db()
load_all_schemas()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
