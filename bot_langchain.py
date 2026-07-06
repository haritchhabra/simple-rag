import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise Exception("GOOGLE_API_KEY not found in .env")

PDF_PATH = "./sample.pdf"
CHROMA_DIR = "chroma_db"

# -----------------------------
# Load embedding model
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Create vector database if needed
# -----------------------------

if not os.path.exists(CHROMA_DIR):

    print("Creating vector database...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"Stored {len(chunks)} chunks.")

else:

    print("Loading existing vector database...")

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

# -----------------------------
# Retriever
# -----------------------------

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}
)

# -----------------------------
# Gemini 2.5 Flash
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# -----------------------------
# Prompt
# -----------------------------

template = """
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

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

print("\n===== PDF RAG Chat =====")

while True:

    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)

    print("\nAnswer:\n")
    print(response.content)

    print("\nSources:")

    for doc in docs:
        print(f"Page {doc.metadata.get('page', 'Unknown') + 1}")