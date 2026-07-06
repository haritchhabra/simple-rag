import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ------------------------
# CONFIG
# ------------------------

PDF_PATH = "./sample.pdf"

client = OpenAI(
    api_key="grok_api_key",
    base_url="https://api.groq.com/openai/v1"
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------
# LOAD PDF
# ------------------------

doc = fitz.open(PDF_PATH)

text = ""
for page in doc:
    text += page.get_text()

# ------------------------
# CHUNKING
# ------------------------

chunk_size = 2000
overlap=200

chunks = []

for i in range(0, len(text), chunk_size-overlap):
    chunks.append(text[i:i+chunk_size])

# ------------------------
# EMBEDDINGS
# ------------------------

embeddings = model.encode(chunks)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# ------------------------
# CHAT LOOP
# ------------------------

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    query_embedding = model.encode([question]).astype("float32")

    D, I = index.search(query_embedding, k=15)

    context = "\n\n".join([chunks[i] for i in I[0]])

    prompt = f"""
You are answering questions about a research paper.

Use ONLY the provided context.

If the answer exists, explain it thoroughly in detail.

If the user asks for a summary, produce a comprehensive summary covering all major sections.

If the answer is not present in the context, clearly state that.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nBot:", response.choices[0].message.content)