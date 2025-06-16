import pandas as pd
from pathlib import Path

# Load the dataset
df = pd.read_csv("sample_data/pii_dataset.csv")

# Show 6 rows from the 'text' column
texts = df['text'].head(10).tolist()
paragraph = " ".join(texts)

file_path = Path("sample_texts.txt")
file_path.write_text(paragraph, encoding="utf-8")