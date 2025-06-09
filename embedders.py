import json
import re
import torch
from enum import Enum
from sentence_transformers import SentenceTransformer
 
class Month(Enum):
    janvier = "Jan"
    février = "Feb"
    mars = "Mar"
    avril = "Apr"
    mai = "May"
    juin = "Jun"
    juillet = "Jul"
    août = "Aug"
    septembre = "Sep"
    octobre = "Oct"
    novembre = "Nov"
    décembre = "Dec"

class ArticleEmbedder():
    def __init__(self, model, article_name):
        self.model = model
        self.txt_path = f"data\\raw\\articles\\{article_name}.txt"

    @staticmethod
    def open_text(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def create_articles_chunks(text):
        pattern = r'\d+\s+mots,\s+p\.\s*\d+'
        matches = re.findall(pattern, text)
        parts = re.split(pattern, text)

        if parts[0].strip() == '':
            parts = parts[1:]

        cleaned_parts = []
        for part in parts:
            lines = [line.strip() for line in part.strip().splitlines() if line.strip()]
            cleaned_parts.append(lines)

        for i in range(len(cleaned_parts) - 1):
            if cleaned_parts[i]:
                last_line = cleaned_parts[i].pop()
                cleaned_parts[i + 1].insert(0, last_line)

        chunks = []
        for i, lines in enumerate(cleaned_parts):
            chunk_text = '\n'.join(lines).strip()
            if chunk_text:
                title = matches[i] if i < len(matches) else ''
                chunk = f"{title}\n{chunk_text}" if title else chunk_text
                chunks.append(chunk)
        chunks.pop(0)
        return chunks
    
    @staticmethod
    def create_dated_chunks(chunks):
        processed = {}

        for i, chunk in enumerate(chunks):
            lines = chunk.splitlines()

            # Extract date line
            if i < len(chunks) - 1:
                line = lines[1].strip()
            else:
                line = lines[0].strip()

            words = line.split()
            month_fr, year = words[-2], words[-1]
            month_en = None

            for month in Month:
                if month.name.lower() == month_fr.lower():
                    month_en = month.value
                    break

            date_str = f"{month_en or month_fr} {year}"

            # Clean chunk text
            if i < len(chunks) - 1:
                modified_chunk = "\n".join(lines[2:])
            else:
                modified_chunk = "\n".join(lines[1:])

            # Create unique key
            chunk_key = f"chunk_{i}"
            processed[chunk_key] = {
                "date": date_str,
                "article": modified_chunk.strip()
            }

        return processed

    
    def create_save_articles_dated_embeddings(self, dated_chunks, article_name):
        keys = list(dated_chunks.keys())
        strings = [dated_chunks[k]["article"] for k in keys]

        embedded_chunks = self.model.encode(strings, convert_to_tensor=True)

        embedded_data = {
            k: {
                "date": dated_chunks[k]["date"],
                "embedding": embedding.tolist()
            }
            for k, embedding in zip(keys, embedded_chunks)
        }

        with open(f'data\\embeddings\\articles\\{article_name}.json', 'w', encoding='utf-8') as f:
            json.dump(embedded_data, f, indent=2, ensure_ascii=False)

    def __call__(self):
        text = self.open_text(self.txt_path)
        articles_chunks = self.create_articles_chunks(text)
        dated_chunks = self.create_dated_chunks(articles_chunks)
        filename = self.txt_path.split('\\')[-1].rsplit('.txt', 1)[0]
        self.create_save_articles_dated_embeddings(dated_chunks, f"{filename} embedding")

class SubjectEmbedder():    
    def __init__(self, model, subject_name):
        self.model = model
        self.txt_path = f"data\\raw\\subjects\\{subject_name}.txt"

    @staticmethod
    def open_text(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    def create_save_subjects_embeddings(self, subject_text, embeddings_name):
        embedded_subject_text = self.model.encode(subject_text, convert_to_tensor=True)
        torch.save(embedded_subject_text, f'data\\embeddings\\subjects\\{embeddings_name}.pt')

    def __call__(self):
        text = self.open_text(self.txt_path)
        filename = self.txt_path.split('\\')[-1].rsplit('.txt', 1)[0]
        self.create_save_subjects_embeddings(text, f"{filename} embedding")
