import re
import torch

class Embedder:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def open_text(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def create_paragraphs_chunks(text):
        chunks = re.split(r'(?<=[.!?])\s*\n', text)
        return [chunk.strip() for chunk in chunks if len(chunk.strip().split()) >= 30]

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
        return chunks

    def create_save_embeddings(self, chunks, type, embeddings_name):
        embedded_chunks = self.model.encode(chunks, convert_to_tensor=True)
        torch.save(embedded_chunks, f'data\\embeddings\\articles\\{embeddings_name}.pt' if type == 0 else f'data\\embeddings\\subjects\\{embeddings_name}.pt' if type == 1 else None)
            
class ArticleEmbedder(Embedder):
    def __init__(self, model, article_name):
        super().__init__(model)
        self.txt_path = f"data\\raw\\articles\\{article_name}.txt"

    def __call__(self):
        text = self.open_text(self.txt_path)
        articles_chunks = self.create_articles_chunks(text)
        filename = self.txt_path.split('\\')[-1].rsplit('.txt', 1)[0]
        self.create_save_embeddings(articles_chunks, 0, f"{filename} embedding")

class SubjectEmbedder(Embedder):
    def __init__(self, model, subject_name):
        super().__init__(model)
        self.txt_path = f"data\\raw\\subjects\\{subject_name}.txt"

    def __call__(self):
        text = self.open_text(self.txt_path)
        filename = self.txt_path.split('\\')[-1].rsplit('.txt', 1)[0]
        self.create_save_embeddings(text, 1, f"{filename} embedding")
