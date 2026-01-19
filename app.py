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
# Safety check: ensure secrets exist
# ----------------------------
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Missing GOOGLE_API_KEY in Streamlit secrets.")
    st.stop()

if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
    st.error("❌ Missing SUPABASE_URL or SUPABASE_KEY in secrets.")
    st.stop()

# Initialize clients
google_api_key = st.secrets["GOOGLE_API_KEY"]
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# ----------------------------
# Optional: S3 loader
# ----------------------------
def load_from_s3(filename):
    if "AWS_ACCESS_KEY_ID" not in st.secrets or "AWS_SECRET_ACCESS_KEY" not in st.secrets:
        st.error("AWS credentials missing in secrets.")
        return None
    s3 = boto3.client(
        's3',
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )
    obj = s3.get_object(Bucket=st.secrets.get("S3_BUCKET", ""), Key=filename)
    return BytesIO(obj['Body'].read())

# ----------------------------
# UI & Logic
# ----------------------------
st.set_page_config(page_title="Gemini Doc Q&A", layout="centered")
st.title("📄 Gemini-Powered Document Q&A")
st.caption("Upload a PDF and ask questions using Google's Gemini model.")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Upload method
use_s3 = st.checkbox("Load from AWS S3 (requires config)")
docs = []

if use_s3:
    s3_key = st.text_input("S3 Object Key (e.g., reports/2024.pdf)")
    if s3_key:
        try:
            file_obj = load_from_s3(s3_key)
            if file_obj:
                loader = PyPDFLoader(file_obj)
                docs = loader.load()
                st.success(f"✅ Loaded from S3: {s3_key}")
        except Exception as e:
            st.error(f"❌ S3 Error: {e}")
else:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
        st.success(f"✅ Loaded {len(docs)} pages from {uploaded_file.name}")

# Process documents
if docs:
    with st.spinner("🧠 Generating embeddings with Gemini... (30-60s)"):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # Use Gemini embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        vectorstore = FAISS.from_documents(splits, embeddings)
        st.session_state.vectorstore = vectorstore

        # Log to Supabase
        filename = s3_key if use_s3 else uploaded_file.name
        supabase.table("documents").insert({"filename": filename}).execute()
        st.info("✅ Ready! Ask a question below.")

# Q&A Interface
if st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    question = st.text_input("❓ Ask a question about your document:")

    if question:
        with st.spinner("🤔 Gemini is thinking..."):
            # Use Gemini LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_api_key,
                temperature=0.3,
                max_output_tokens=1024
            )

            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer based ONLY on the context below. "
                "If you don't know, say 'I cannot answer based on the provided document.'\n\n"
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

                # Log to Supabase
                supabase.table("chat_history").insert({
                    "session_id": "gemini_demo",
                    "question": question,
                    "answer": answer
                }).execute()
            except Exception as e:
                st.error(f"❌ Gemini error: {e}")
else:
    st.info("👆 Upload a PDF or enter an S3 key to begin.")
