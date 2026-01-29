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

# BIẾN TOÀN CỤC: 
# Thay vì lưu 1 chuỗi string khổng lồ, ta lưu dạng danh sách để tìm kiếm
GLOBAL_SCHEMA_DOCS = []  # Chứa chi tiết từng bảng/hàm
GLOBAL_TABLE_NAMES = []  # Chứa danh sách tên rút gọn

# Giới hạn Token an toàn (ước lượng ký tự) để không bị lỗi 400
MAX_CONTEXT_CHARS = 50000 

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
#  PHẦN 4: KỸ THUẬT RAG & LOADING
# =========================================================
def load_all_schemas():
    """
    Load schemas vào bộ nhớ nhưng chia nhỏ thành list để tìm kiếm (Retrieval)
    thay vì gộp tất cả thành 1 cục text khổng lồ.
    """
    global GLOBAL_SCHEMA_DOCS, GLOBAL_TABLE_NAMES
    GLOBAL_SCHEMA_DOCS = []
    GLOBAL_TABLE_NAMES = []
    
    print("🚀 Đang nạp Schemas vào bộ nhớ (Indexing mode)...")
    
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
                    doc_content = ""
                    search_text = ""
                    doc_name = ""
                    
                    # --- XỬ LÝ TABLE ---
                    if 'table_name' in item:
                        name = item.get('table_name', 'Unknown')
                        ddl = item.get('ddl', '')
                        doc_name = name
                        GLOBAL_TABLE_NAMES.append(name)
                        
                        doc_content = f"""
[TABLE SCHEMA]
Name: `{name}`
DDL:
```sql
{ddl}
```
"""
                        # Text dùng để search keyword
                        search_text = (name + " " + ddl).lower()
                    
                    # --- XỬ LÝ ROUTINE ---
                    elif 'routine_name' in item:
                        name = item.get('routine_name', 'Unknown')
                        routine_def = item.get('routine_definition', '')
                        ddl = item.get('ddl', '')
                        arguments = item.get('arguments', [])
                        doc_name = name
                        
                        args_str = json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, (list, dict)) else str(arguments)
                        
                        doc_content = f"""
[ROUTINE / FUNCTION]
Name: `{name}`
Arguments: {args_str}
Routine Definition:
```sql
{routine_def}
```
DDL:
```sql
{ddl}
```
(AI NOTE: Hãy đọc kỹ code SQL trên. Nếu có CASE WHEN, hãy dùng nó để map giá trị ID tương ứng)
"""
                        # Text dùng để search keyword
                        search_text = (name + " " + routine_def + " " + ddl).lower()

                    if doc_content:
                        GLOBAL_SCHEMA_DOCS.append({
                            "name": doc_name,
                            "content": doc_content,
                            "search_text": search_text
                        })

        except Exception as e:
            print(f"❌ Lỗi đọc file {file_path}: {e}")

    print(f"✅ Đã index xong {len(GLOBAL_SCHEMA_DOCS)} đối tượng schema.")

def get_relevant_schemas(user_msg):
    """
    Hàm tìm kiếm thông minh: Chỉ lấy những Schema có liên quan đến câu hỏi.
    Giải quyết vấn đề 'Prompt too long'.
    """
    if not GLOBAL_SCHEMA_DOCS:
        return "No schema data loaded."
    
    query_tokens = user_msg.lower().split()
    scored_docs = []
    
    # 1. Chấm điểm sự liên quan
    for doc in GLOBAL_SCHEMA_DOCS:
        score = 0
        for token in query_tokens:
            # Nếu từ khóa xuất hiện trong tên bảng hoặc nội dung DDL -> tăng điểm
            if token in doc['search_text']:
                score += 1
        scored_docs.append((score, doc))
    
    # 2. Sắp xếp: Điểm cao lên đầu
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Chọn lọc: Lấy top docs sao cho không quá giới hạn ký tự
    selected_contents = []
    current_chars = 0
    
    # Luôn lấy ít nhất top 5 bảng liên quan nhất, hoặc nhiều hơn nếu còn dư chỗ
    for score, doc in scored_docs:
        # Lấy những bảng có match keyword (score > 0) 
        # Hoặc lấy tối thiểu 3 bảng đầu tiên nếu không match gì cả (để AI không bị mù)
        if score > 0 or len(selected_contents) < 3:
            if current_chars + len(doc['content']) < MAX_CONTEXT_CHARS:
                selected_contents.append(doc['content'])
                current_chars += len(doc['content'])
            else:
                break # Đã đầy bộ nhớ context cho phép
    
    return "\n----------------------------------------\n".join(selected_contents)

# --- KHỞI CHẠY LẦN ĐẦU ---
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
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    rows = c.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id): 
    return jsonify(get_chat_history_formatted(session_id, limit=50))

@app.route('/api/chat', methods=['POST'])
def chat():
    # Sử dụng biến toàn cục
    global GLOBAL_TABLE_NAMES
    
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

        # 2. LẤY CONTEXT LIÊN QUAN (RAG)
        # Thay vì đưa toàn bộ, chỉ đưa những gì cần thiết
        relevant_schema_context = get_relevant_schemas(user_msg)
        all_tables_list = ", ".join(GLOBAL_TABLE_NAMES)

        # 3. XÂY DỰNG PROMPT
        system_prompt = f"""Bạn là một chuyên gia BigQuery SQL cao cấp.

[DANH SÁCH TOÀN BỘ CÁC BẢNG HIỆN CÓ]:
{all_tables_list}

[CHI TIẾT SCHEMA & HÀM LIÊN QUAN ĐẾN CÂU HỎI]:
(Hệ thống đã tự động lọc bớt các bảng không liên quan để tối ưu bộ nhớ)
{relevant_schema_context}

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
        history = get_chat_history_formatted(session_id, limit=10)
        for msg in history:
            if msg['content'] != user_msg: 
                messages_payload.append(msg)
        
        messages_payload.append({"role": "user", "content": user_msg})

        # 4. Gọi AI
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        
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
    return jsonify({"status": "success", "message": "Đã nạp lại và index dữ liệu Schema!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

