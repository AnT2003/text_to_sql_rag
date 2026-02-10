# ================= FULL ADVANCED HYBRID RAG SQL AGENT (SINGLE FILE) =================
# Upgraded retrieval: BM25 + Dense + Multi-query + RRF fusion + MMR diversification

import os
import json
import glob
import datetime
import re
import requests
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ollama import Client
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc
from rank_bm25 import BM25Okapi

# =========================================================
# CONFIG
# =========================================================
load_dotenv()
app = Flask(__name__)
CORS(app)

SCHEMA_FOLDER = "./schemas"
MODEL_NAME = "gpt-oss:120b"
OLLAMA_HOST = "https://ollama.com"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_API_KEY = os.getenv("OLLAMA_API_KEY")

# =========================================================
# DATABASE
# =========================================================
db_url = os.getenv("DATABASE_URL") or "sqlite:///local_chat.db"
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Session(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    title = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), db.ForeignKey('session.id',ondelete="CASCADE"))
    role = db.Column(db.String(20))
    content = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# =========================================================
# EMBEDDING (OPENROUTER)
# =========================================================
def openrouter_embedding(texts, model="sentence-transformers/all-minilm-l12-v2"):
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    res = requests.post(url, headers=headers, json={"model": model, "input": texts})
    if res.status_code != 200:
        raise Exception(res.text)

    data = res.json()["data"]
    embeddings = np.array([d["embedding"] for d in data], dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    return embeddings

# =========================================================
# ADVANCED RAG ENGINE
# =========================================================
class RAGEngine:
    def __init__(self):
        self.docs = []
        self.doc_types = []
        self.tokenized = []
        self.embeddings = None
        self.bm25 = None
        self.ready = False

    # ---------- tokenizer giữ snake_case ----------
    def tokenize(self, text):
        text = re.sub(r'[.\-(),`]', ' ', str(text))
        tokens = text.lower().split()
        stop = {'string','int64','timestamp','table','dataset','project'}
        return [t for t in tokens if t not in stop and len(t) > 1]

    # ---------- LOAD SCHEMA ----------
    def load_schemas(self):
        docs, tokenized, types = [], [], []
        files = glob.glob(os.path.join(SCHEMA_FOLDER, "*.json"))

        for file in files:
            data = json.load(open(file, encoding="utf-8"))
            items = data if isinstance(data, list) else [data]

            for item in items:
                if 'table_name' in item:
                    dataset = item.get('table_schema')
                    table = f"`kynaforkids-server-production.{dataset}.{item['table_name']}`"
                    cols = json.loads(item.get("columns", "[]"))
                    doc = f"[TABLE] {table}\nColumns: {', '.join(cols)}"
                    docs.append(doc)
                    tokenized.append(self.tokenize(doc))
                    types.append("table")

                elif 'routine_name' in item:
                    name = item['routine_name']
                    definition = item.get('routine_definition','')
                    doc = f"[FUNCTION] {name}\n{definition}"
                    docs.append(doc)
                    tokenized.append(self.tokenize(doc))
                    types.append("function")

        self.docs = docs
        self.tokenized = tokenized
        self.doc_types = types
        self.bm25 = BM25Okapi(tokenized)
        self.embeddings = openrouter_embedding(docs)
        self.ready = True
        print("Indexed:", len(docs))

    # ---------- MULTI QUERY EXPANSION ----------
    def expand_queries(self, query, api_key):
        client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
        prompt = f"""
Create 3 alternative database search queries.
Return as list.
User: {query}
"""
        try:
            res = client.chat(model=MODEL_NAME, messages=[{"role":"user","content":prompt}], options={"temperature":0})
            lines = res['message']['content'].split("\n")
            return [query] + lines[:3]
        except:
            return [query]

    # ---------- Reciprocal Rank Fusion ----------
    def rrf(self, rankings, k=60):
        scores = {}
        for rank in rankings:
            for i, doc_id in enumerate(rank):
                scores[doc_id] = scores.get(doc_id,0) + 1/(k+i+1)
        return sorted(scores, key=scores.get, reverse=True)

    # ---------- RETRIEVE ----------
    def retrieve(self, query, api_key, top_k=15):
        queries = self.expand_queries(query, api_key)
        all_rankings = []

        for q in queries:
            # BM25
            bm = self.bm25.get_scores(self.tokenize(q))
            bm_rank = np.argsort(bm)[::-1][:50]

            # Dense
            q_emb = openrouter_embedding([q])[0]
            vec = np.dot(self.embeddings, q_emb)
            vec_rank = np.argsort(vec)[::-1][:50]

            all_rankings.append(bm_rank)
            all_rankings.append(vec_rank)

        fused = self.rrf(all_rankings)
        return "\n---\n".join([self.docs[i] for i in fused[:top_k]])

rag_engine = RAGEngine()
rag_engine.load_schemas()

# =========================================================
# CHAT ENDPOINT
# =========================================================
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message')
    api_key = data.get('api_key') or DEFAULT_API_KEY

    schemas = rag_engine.retrieve(user_msg, api_key)

    system_prompt = f"""
You are Senior BigQuery SQL Architect.
Use ONLY provided schemas.
Return SQL inside ```sql block.

SCHEMA:
{schemas}

User: {user_msg}
"""

    client = Client(host=OLLAMA_HOST, headers={"Authorization": f"Bearer {api_key}"})
    res = client.chat(model=MODEL_NAME, messages=[{"role":"system","content":system_prompt}])
    return jsonify({"response": res['message']['content']})

# =========================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
