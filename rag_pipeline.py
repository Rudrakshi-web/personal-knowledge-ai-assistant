from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# -----------------------------
# LOAD VECTOR DATABASE
# -----------------------------
def load_vector_db():
    print("🧠 Loading vector database...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory="vector_db",
        embedding_function=embedding_model
    )

    print("✅ Vector DB loaded!")
    return vector_db


# -----------------------------
# CREATE RETRIEVER
# -----------------------------
def get_retriever(vector_db):
    retriever = vector_db.as_retriever(search_kwargs={"k":2})
    return retriever


# -----------------------------
# LOAD LOCAL LLM
# -----------------------------
print("🧠 Loading TinyLlama model...")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

print("✅ Model ready!")


# -----------------------------
# LOAD VECTOR DB ONCE (FAST MODE)
# -----------------------------
vector_db = load_vector_db()
retriever = get_retriever(vector_db)


# -----------------------------
# ASK ASSISTANT FUNCTION
# -----------------------------
def ask_assistant(question):

    docs = retriever.invoke(question)

    # ---- Controlled Context ----
    context_parts = []
    current_length = 0
    MAX_CONTEXT = 1200

    for doc in docs:
        text = doc.page_content.strip()
        if current_length + len(text) > MAX_CONTEXT:
            break
        context_parts.append(text)
        current_length += len(text)

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a study assistant.

Use ONLY the CONTEXT to answer.

If answer is not found, say:
Not clearly defined in the provided material.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔥 ULTRA CLEAN OUTPUT
    if "[/INST]" in response:
        answer = response.split("[/INST]")[-1]
    else:
        answer = response

    answer = answer.replace("<|assistant|>", "")
    answer = answer.replace("<s>", "")
    answer = answer.replace("</s>", "")
    answer = answer.strip()

    return answer
