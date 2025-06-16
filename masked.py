# --- masked.py ---

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from faker import Faker
import re
import uuid
import torch

# Load detection models and Faker
def load_pipelines():
    fake = Faker()
    device = torch.device("cpu")

    pii_model_name = "iiiorg/piiranha-v1-detect-personal-information"
    pii_tokenizer = AutoTokenizer.from_pretrained(pii_model_name)
    pii_model = AutoModelForTokenClassification.from_pretrained(pii_model_name).to(device)
    pii_detector = pipeline("token-classification", model=pii_model, tokenizer=pii_tokenizer, device=-1, aggregation_strategy="simple")

    name_model = "Jean-Baptiste/roberta-large-ner-english"
    name_detector = pipeline("token-classification", model=name_model, tokenizer=name_model, device=-1, aggregation_strategy="simple")

    return pii_detector, name_detector, fake

# Smart dummy generators
def get_smart_dummy(label, fake):
    fake = Faker()
    if label == "EMAIL":
        return f"user.{uuid.uuid4().hex[:5]}@maskmail.com"
    elif label == "TELEPHONENUM":
        return fake.phone_number()
    elif label == "SSN":
        return f"{fake.random_number(3):03d}-{fake.random_number(2):02d}-{fake.random_number(4):04d}"
    elif label == "CREDITCARDNUMBER":
        return fake.credit_card_number()
    elif label == "ACCOUNTNUM":
        return f"{fake.random_number(9):09d}"
    elif label == "DATEOFBIRTH":
        return fake.date_of_birth().isoformat()
    elif label == "ZIPCODE":
        return fake.zipcode()
    elif label == "ADDRESS":
        return fake.address().replace("\n", ", ")
    elif label == "NAME":
        return fake.name()
    else:
        return f"[MASKED-{label}]"

# Mask a text string
def mask_text(text, pii_detector, name_detector, fake, supabase=None):
    if not isinstance(text, str) or not text.strip():
        return text

    pii_results = pii_detector(text)
    name_results = name_detector(text)

    combined_results = pii_results + [
        {**ent, "entity_group": "NAME"}
        for ent in name_results
        if ent["entity_group"] in ("PER", "B-PER", "I-PER")
    ]

    replacements = {}
    for ent in combined_results:
        word = ent["word"]
        label = ent["entity_group"]
        if word not in replacements:
            masked = get_smart_dummy(label, fake)
            replacements[word] = masked
            if supabase:
                supabase.table("pii_map").insert({"original": word, "masked_text": masked}).execute()

    sorted_replacements = sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True)

    masked_text = text
    for original, replacement in sorted_replacements:
        pattern = re.escape(original)
        masked_text = re.sub(pattern, replacement, masked_text)

    return masked_text

# Mask an entire dataframe using column classification
def mask_watsonx_dataframe(df, pii_detector, name_detector, fake, supabase, column_labels):
    masked_df = df.copy()
    for col in df.columns:
        label = column_labels.get(col, "NON-SENSITIVE")
        if label != "NON-SENSITIVE":
            masked_values = []
            for val in df[col].astype(str):
                fake_value = get_smart_dummy(label, fake)
                supabase.table("pii_map").insert({"original": val, "masked_text": fake_value}).execute()
                masked_values.append(fake_value)
            masked_df[col] = masked_values
    return masked_df

# Mask generic text df (e.g. .txt)
def mask_dataframe(df, pii_detector, name_detector, fake, supabase=None):
    return df.applymap(lambda cell: mask_text(cell, pii_detector, name_detector, fake, supabase))
