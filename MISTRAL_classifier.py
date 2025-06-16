from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_text(prompt, tokenizer, model, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy decoding
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_column_label_from_local_model(column_name, samples, tokenizer, model):
    sample_text = ", ".join(map(str, samples[:3]))
    prompt = f"""
You are a data privacy expert. Your task is to determine the most likely personal data category (PII) for a dataset column, based on the column name and sample values.

Here are the possible labels:
- NAME
- EMAIL
- TELEPHONENUM
- SSN
- ADDRESS
- CREDITCARDNUMBER
- ACCOUNTNUM
- DATEOFBIRTH
- ZIPCODE
- NON-SENSITIVE

Column: {column_name}
Sample values: {sample_text}

Respond with just one label from the above list.
""".strip()

    response = generate_text(prompt, tokenizer, model)
    return response.strip().split('\n')[-1].strip().upper()

def classify_all_columns_with_local_model(df, tokenizer, model):
    column_labels = {}
    for col in df.columns:
        values = df[col].dropna().astype(str).tolist()
        if not values:
            column_labels[col] = "NON-SENSITIVE"
            continue
        label = get_column_label_from_local_model(col, values, tokenizer, model)
        column_labels[col] = label
    return column_labels