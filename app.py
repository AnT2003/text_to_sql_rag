import os
import json
import glob
import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client

# --- 1. SETUP MÔI TRƯỜNG ---
load_dotenv()
app = Flask(__name__)

# --- 2. CẤU HÌNH HỆ THỐNG ---
SCHEMA_FOLDER = "./schemas"

# Cấu hình Database (PostgreSQL) - Lấy từ biến môi trường
# Trên Render/Heroku, biến này sẽ tự động được cung cấp
DATABASE_URL = os.getenv("DATABASE_URL")

# Cấu hình Ollama (Cloud hoặc Local)
OLLAMA_HOST = "https://ollama.com"
MODEL_NAME = "gpt-oss:120b"
# API Key (Ưu tiên lấy từ .env)
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY") 

# BIẾN TOÀN CỤC: Chứa danh sách các Documents (Chunks) để làm RAG
GLOBAL_SCHEMA_DOCS = []

# =========================================================
#  PHẦN 3: QUẢN LÝ DATABASE (POSTGRESQL) - LƯU LỊCH SỬ CHAT
# =========================================================
def get_db_connection():
    """Tạo kết nối đến PostgreSQL"""
    if not DATABASE_URL:
        print("❌ Lỗi: Chưa cấu hình DATABASE_URL trong .env hoặc biến môi trường!")
        return None
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
        print("✅ Đã khởi tạo Database PostgreSQL thành công.")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Database: {e}")

def get_chat_history_formatted(session_id, limit=10):
    """Lấy lịch sử chat của một phiên cụ thể"""
    conn = get_db_connection()
    if not conn: return []
    
    try:
        # Sử dụng RealDictCursor để lấy dữ liệu dạng Dictionary
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Postgres dùng %s thay vì ? cho tham số
        cur.execute("SELECT role, content FROM messages WHERE session_id = %s ORDER BY created_at DESC LIMIT %s", (session_id, limit))
        rows = cur.fetchall()
        
        conn.close()
        
        history = []
        # Đảo ngược để xếp theo thứ tự thời gian cũ -> mới
        for r in rows[::-1]:
            history.append({"role": r["role"], "content": r["content"]})
        return history
    except Exception as e:
        print(f"Lỗi lấy lịch sử: {e}")
        return []

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

# =========================================================
#  PHẦN 4: KỸ THUẬT RAG (RETRIEVAL AUGMENTED GENERATION)
# =========================================================
def load_all_schemas():
    """
    Kỹ thuật Advanced: Đọc TẤT CẢ file schemas và Indexing cho RAG.
    Thay vì gộp thành 1 chuỗi, ta lưu thành từng mảnh (document) để tìm kiếm.
    """
    global GLOBAL_SCHEMA_DOCS
    print("🚀 Đang nạp Schemas và Indexing cho RAG...")
    
    if not os.path.exists(SCHEMA_FOLDER): 
        print(f"⚠️ Không tìm thấy thư mục {SCHEMA_FOLDER}")
        return

    json_files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))
    GLOBAL_SCHEMA_DOCS = [] # Reset list
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    # --- XỬ LÝ TABLE (BẢNG) ---
                    if 'table_name' in item:
                        name = item.get('table_name', 'Unknown')
                        ddl = item.get('ddl', '')
                        
                        doc_content = f"""
[TABLE SCHEMA]
Name: `{name}`
DDL:
```sql
{ddl}
```
"""
                        GLOBAL_SCHEMA_DOCS.append({
                            "name": name,
                            "type": "TABLE",
                            "content": doc_content,
                            "keywords": f"{name} {ddl}".lower() # Index keywords
                        })
                    
                    # --- XỬ LÝ ROUTINE (HÀM) ---
                    elif 'routine_name' in item:
                        name = item.get('routine_name', 'Unknown')
                        ddl = item.get('ddl', '')
                        definition = item.get('routine_definition', '')
                        arguments = item.get('arguments', [])
                        
                        # Format arguments
                        if isinstance(arguments, (list, dict)):
                            args_str = json.dumps(arguments, ensure_ascii=False)
                        else:
                            args_str = str(arguments)
                        
                        code_content = ddl if ddl else definition
                        
                        doc_content = f"""
[ROUTINE / FUNCTION]
Name: `{name}`
Arguments: {args_str}
DEFINITION (SOURCE SQL CODE):
```sql
{code_content}
```
(AI NOTE: Hãy đọc kỹ code SQL trên. Nếu có CASE WHEN, hãy dùng nó để map giá trị ID tương ứng)
"""
                        GLOBAL_SCHEMA_DOCS.append({
                            "name": name,
                            "type": "ROUTINE",
                            "content": doc_content,
                            "keywords": f"{name} {code_content}".lower()
                        })

        except Exception as e:
            print(f"❌ Lỗi đọc file {file_path}: {e}")

    print(f"✅ Đã nạp {len(GLOBAL_SCHEMA_DOCS)} documents vào bộ nhớ RAG.")

def search_relevant_schemas(query, top_k=10):
    """
    Hàm RAG Retrieval: Tìm kiếm schema liên quan dựa trên từ khóa.
    """
    if not GLOBAL_SCHEMA_DOCS:
        return []
    
    query_lower = query.lower()
    query_tokens = set(query_lower.split())
    
    scored_docs = []
    
    for doc in GLOBAL_SCHEMA_DOCS:
        score = 0
        doc_keywords = doc['keywords']
        
        # 1. Ưu tiên khớp tên bảng/hàm (Trọng số cao)
        if doc['name'].lower() in query_lower:
            score += 20
            
        # 2. Khớp từng từ khóa
        for token in query_tokens:
            if len(token) > 2 and token in doc_keywords:
                score += 1
        
        if score > 0:
            scored_docs.append((score, doc['content']))
    
    # Sắp xếp theo điểm giảm dần
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Lấy top K kết quả
    relevant_chunks = [item[1] for item in scored_docs[:top_k]]
    
    # Fallback: Trả về một ít nếu không tìm thấy gì để AI không bị mù
    if not relevant_chunks and GLOBAL_SCHEMA_DOCS:
        return [doc['content'] for doc in GLOBAL_SCHEMA_DOCS[:3]]
        
    return relevant_chunks

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
    
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM sessions ORDER BY created_at DESC")
        rows = cur.fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        print(f"Lỗi lấy danh sách session: {e}")
        return jsonify([])

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id): 
    """API lấy nội dung chat"""
    return jsonify(get_chat_history_formatted(session_id, limit=50))

@app.route('/api/chat', methods=['POST'])
def chat():
    # Sử dụng logic RAG thay vì Global Full Schema
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

        # 2. RAG RETRIEVAL: Tìm các Schema liên quan
        print(f"🔍 Đang tìm schema liên quan cho câu hỏi: {user_msg}")
        relevant_schemas = search_relevant_schemas(user_msg, top_k=8)
        
        rag_context = "\n----------------------------------------\n".join(relevant_schemas)
        if not rag_context:
            rag_context = "(Không tìm thấy bảng nào khớp rõ rệt. Hãy dùng kiến thức SQL chung.)"

        # 3. XÂY DỰNG PROMPT (Với context đã được lọc gọn)
        system_prompt = f"""Bạn là một chuyên gia BigQuery SQL cao cấp.

[RAG CONTEXT - DỮ LIỆU LIÊN QUAN NHẤT]:
Hệ thống đã tự động lọc ra các Bảng và Hàm có khả năng liên quan đến câu hỏi của user.
Chỉ sử dụng thông tin này để viết query:

{rag_context}

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
        
        # Thêm lịch sử chat gần nhất
        history = get_chat_history_formatted(session_id, limit=5)
        for msg in history:
            if msg['content'] != user_msg: 
                messages_payload.append(msg)
        
        # Thêm câu hỏi hiện tại
        messages_payload.append({"role": "user", "content": user_msg})

        # 4. Gọi AI
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        
        try:
            response = client.chat(
                model=MODEL_NAME, 
                messages=messages_payload, 
                stream=False, 
                options={"temperature": 0.1}
            )
            ai_reply = response['message']['content']
        except Exception as ollama_error:
            # Xử lý lỗi token limit nếu vẫn bị
            err_msg = str(ollama_error)
            print(f"⚠️ Lỗi gọi AI: {err_msg}")
            
            if "too long" in err_msg or "400" in err_msg:
                print("⚠️ Context vẫn dài, thử lại với ít schema hơn...")
                less_relevant = search_relevant_schemas(user_msg, top_k=3)
                less_context = "\n".join(less_relevant)
                
                messages_payload[0]['content'] = system_prompt.replace(rag_context, less_context)
                
                response = client.chat(
                    model=MODEL_NAME, 
                    messages=messages_payload, 
                    stream=False, 
                    options={"temperature": 0.1}
                )
                ai_reply = response['message']['content']
            else:
                return jsonify({"error": f"Lỗi AI: {err_msg}"}), 500

        # 5. Lưu câu trả lời của AI
        save_message(session_id, "assistant", ai_reply)

        return jsonify({"response": ai_reply})

    except Exception as e:
        print(f"Lỗi Server: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reload', methods=['POST'])
def reload_schema():
    """API để nạp lại dữ liệu khi bạn sửa file JSON"""
    load_all_schemas()
    return jsonify({"status": "success", "message": "Đã nạp lại và re-index dữ liệu cho RAG!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
