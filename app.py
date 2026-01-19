import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from supabase import create_client
import boto3
from io import BytesIO

# ----------------------------
# Initialize from secrets
# ----------------------------
def get_required_secret(key: str):
    if key not in st.secrets:
        st.error(f"❌ Missing required secret: {key}")
        st.stop()
    return st.secrets[key]

GOOGLE_API_KEY = get_required_secret("GOOGLE_API_KEY")
SUPABASE_URL = get_required_secret("SUPABASE_URL")
SUPABASE_KEY = get_required_secret("SUPABASE_KEY")

# Initialize Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# Optional S3 loader
# ----------------------------
def load_from_s3(filename):
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        obj = s3.get_object(Bucket=st.secrets["S3_BUCKET"], Key=filename)
        return BytesIO(obj['Body'].read())
    except Exception as e:
        st.error(f"❌ S3 error: {str(e)}")
        return None

# ----------------------------
# Safe Supabase logging (won't crash app)
# ----------------------------
def log_to_supabase(table: str, data: dict):
    try:
        supabase.table(table).insert(data).execute()
    except Exception as e:
        # Silent fail for demo — remove in production
        pass

# ----------------------------
# Main App
# ----------------------------
st.set_page_config(page_title="Gemini Doc Q&A", layout="centered")
st.title("📄 Gemini-Powered Document Q&A")
st.caption("Upload a PDF and ask questions using Google's Gemini model.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Upload method
use_s3 = st.checkbox("Load from AWS S3")
docs = []

if use_s3:
    if "AWS_ACCESS_KEY_ID" not in st.secrets or "S3_BUCKET" not in st.secrets:
        st.warning("⚠️ AWS credentials or S3_BUCKET not configured in secrets.")
    else:
        s3_key = st.text_input("S3 Object Key (e.g., docs/manual.pdf)")
        if s3_key:
            file_obj = load_from_s3(s3_key)
            if file_obj:
                loader = PyPDFLoader(file_obj)
                docs = loader.load()
                st.success(f"✅ Loaded from S3: {s3_key}")
else:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
        st.success(f"✅ Loaded {len(docs)} pages")

# Process documents
if docs:
    with st.spinner("🧠 Generating embeddings with Gemini... (30-60s)"):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        vectorstore = FAISS.from_documents(splits, embeddings)
        st.session_state.vectorstore = vectorstore

        # Log filename (safe)
        filename = s3_key if use_s3 and 's3_key' in locals() else (uploaded_file.name if not use_s3 else "unknown")
        log_to_supabase("documents", {"filename": filename})
        st.info("✅ Ready! Ask a question below.")

# Q&A
if st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    question = st.text_input("❓ Ask a question about your document:")

    if question:
        with st.spinner("🤔 Gemini is thinking..."):
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                max_output_tokens=1024
            )

            prompt = ChatPromptTemplate.from_template(
                "Answer based ONLY on the context below. If unsure, say 'I cannot answer based on the provided document.'\n\n"
                "Context:\n{context}\n\nQuestion: {question}"
            )

            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            try:
                answer = chain.invoke(question)
                st.write("**💡 Answer:**", answer)
                log_to_supabase("chat_history", {
                    "session_id": "demo",
                    "question": question,
                    "answer": answer
                })
            except Exception as e:
                st.error(f"❌ Gemini error: {str(e)}")
else:
    st.info("👆 Upload a PDF or enable S3 to begin.")
