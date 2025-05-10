from flask import Flask, render_template, request
from transformers import AutoTokenizer, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import torch
import os

# === Device setup ===
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# === Load model and tokenizer ===
model_name = "PY007/TinyLlama-1.1B-Chat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    device=device,
    torch_dtype=dtype,
    trust_remote_code=True
)

# === Load and split PDFs ===
def load_and_split_pdfs(pdf_folder):
    pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks

# === Create FAISS vector store ===
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def create_faiss_index(chunks):
    return FAISS.from_documents(chunks, embedding_model)

def retrieve_context(query, index, k=3):
    docs = index.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

# === Build or Load FAISS index ===
vectorstore_dir = "vectorstore/"
if os.path.exists(os.path.join(vectorstore_dir, "index.faiss")):
    faiss_index = FAISS.load_local(vectorstore_dir, embedding_model)
else:
    chunks = load_and_split_pdfs("docs")
    faiss_index = create_faiss_index(chunks)
    faiss_index.save_local(vectorstore_dir)

# === Flask App ===
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    user_input = ""

    if request.method == "POST":
        user_input = request.form["question"].strip()
        if user_input:
            try:
                context = retrieve_context(user_input, faiss_index)
                prompt = f"""Summarize the following content or explain it in simple terms. 

{context}

### Human: {user_input}
### Assistant:"""
                result = generator(
                    prompt,
                    max_new_tokens=300,
                    do_sample=True,
                    top_k=50,
                    top_p=0.7,
                    repetition_penalty=1.1
                )
                response = result[0]['generated_text'].split("### Assistant:")[-1].strip()
            except Exception as e:
                response = f"Error: {e}"

    return render_template("index.html", response=response, user_input=user_input)

if __name__ == "__main__":
    app.run(debug=True)
