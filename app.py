import time
from flask import Flask, request, render_template, jsonify
from transformers import pipeline, AutoTokenizer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import torch
import os
import re

app = Flask(__name__)

# === Model Setup with Load Timer ===
start_time = time.time()

model_name = "PY007/TinyLlama-1.1B-Chat-v0.1"
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    device=device,
    torch_dtype=dtype,
    trust_remote_code=True
)

load_time = time.time() - start_time
print(f"✅ Model loaded in {load_time:.2f} seconds")

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore_dir = "vectorstore"
docs_dir = "docs"

def load_or_build_index():
    if os.path.exists(os.path.join(vectorstore_dir, "index.faiss")):
        return FAISS.load_local(
            vectorstore_dir,
            embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        print("📥 No FAISS index found. Loading all PDFs from ./docs/")
        pdf_paths = [os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if f.endswith(".pdf")]

        documents = []
        for path in pdf_paths:
            print(f"📄 Loading: {path}")
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        print(f"🧩 Total Chunks: {len(chunks)}")

        index = FAISS.from_documents(chunks, embedding_model)
        os.makedirs(vectorstore_dir, exist_ok=True)
        index.save_local(vectorstore_dir)
        print("✅ Vector index saved.")
        return index

faiss_index = load_or_build_index()

def highlight_response_in_context(response, context):
    escaped = re.escape(response[:50])
    pattern = re.compile(f"({escaped})", re.IGNORECASE)
    return pattern.sub(r'<mark>\1</mark>', context)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"response": "Please enter a question.", "evidence": ""})

    try:
        query_start = time.time()

        docs = faiss_index.similarity_search(user_input, k=3)
        context = "\n---\n".join([doc.page_content for doc in docs])

        prompt = f"""Use the following context to answer the user's question.

Context:
{context}

### Human: {user_input}
### Assistant:"""

        result = generator(
            prompt,
            max_new_tokens=300,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.1
        )

        answer = result[0]['generated_text'].split("### Assistant:")[-1].strip()
        highlighted_context = highlight_response_in_context(answer, context)

        query_end = time.time()
        print(f"⏱️ Response time: {query_end - query_start:.2f} seconds")

        return jsonify({
            "response": answer,
            "evidence": highlighted_context
        })
    except Exception as e:
        return jsonify({"response": f"Error: {e}", "evidence": ""})

if __name__ == "__main__":
    app.run(debug=True)
