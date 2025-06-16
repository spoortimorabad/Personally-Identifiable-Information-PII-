import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from masked import load_pipelines, mask_dataframe, mask_watsonx_dataframe
from MISTRAL_classifier import classify_all_columns_with_local_model  # NEW local classifier
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import login

login("Your_HUGGING_FACE_TOKEN")

# To avoid rerun triggers on file system changes
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

# Streamlit title
st.title("🔐 PII Detection & Masking Tool")

# Load environment variables
load_dotenv()

# Load Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load PII detection and faker pipelines
pii_detector, name_detector, fake = load_pipelines()

# Load Hugging Face mistral model
@st.cache_resource
def load_local_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_local_model()

# Helper: Insert mapping into Supabase
def insert_pii_mapping(original, masked):
    supabase.table("pii_map").insert({"original": original, "masked_text": masked}).execute()

# File upload
uploaded_file = st.file_uploader("📂 Upload your dataset (.csv or .txt)", type=["csv", "txt"])

if uploaded_file:
    file_name = uploaded_file.name

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Original CSV Data")
        st.dataframe(df)

        if st.button("🚀 Classify and Mask CSV PII"):
            with st.spinner("🔍 Classifying sensitive columns..."):
                column_labels = classify_all_columns_with_local_model(df, tokenizer, model)

            st.subheader("🧠 Column Classifications")
            for col, label in column_labels.items():
                st.markdown(f"{col}** → {label}")

            with st.spinner("🔐 Masking sensitive values..."):
                masked_df = mask_watsonx_dataframe(
                    df, pii_detector, name_detector, fake, supabase, column_labels
                )

            st.subheader("✅ Masked CSV Data")
            st.dataframe(masked_df)

            csv = masked_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Masked CSV", csv, "masked_dataset.csv", "text/csv")

    elif file_name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8")
        paragraphs = [para.strip() for para in content.split("\n\n") if para.strip()]
        df = pd.DataFrame(paragraphs, columns=["text"])

        st.subheader("📄 Original Text Content")
        st.dataframe(df)

        if st.button("🚀 Mask Text PII"):
            with st.spinner("🔐 Masking text using local pipeline..."):
                column_labels = {"text": "TEXT"}
                masked_df = mask_dataframe(df, pii_detector, name_detector, fake, supabase)

            st.subheader("✅ Masked Text")
            st.dataframe(masked_df)

            csv = masked_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Masked TXT", csv, "masked_text.csv", "text/csv")