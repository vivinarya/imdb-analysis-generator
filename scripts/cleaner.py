import pandas as pd
import re

def clean_imdb_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'--.*', '', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text


df = pd.read_csv('IMDB Dataset.csv', encoding='utf-8')

print(f'Dataset loaded: {len(df)} reviews')
print(f'Columns: {list(df.columns)}')


df['cleaned_review'] = df['review'].apply(clean_imdb_text)

df = df[df['cleaned_review'].str.len() > 10]


df[['cleaned_review', 'sentiment']].to_csv('imdb_cleaned_.csv', index=False)
print('Cleaned dataset saved as imdb_cleaned.csv')
print('Ready for model training!')
