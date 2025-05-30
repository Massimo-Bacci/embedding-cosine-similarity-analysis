import os
from sentence_transformers import SentenceTransformer
from embedders import ArticleEmbedder, SubjectEmbedder

def embed_all_files(embedder_class, directory, model, label):
    print(f"\n🔍 Starting {label.lower()} embedding from directory: {directory}")
    txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]

    if not txt_files:
        print(f"⚠️  No .txt files found in {directory}")
        return

    for file in txt_files:
        name = os.path.splitext(file)[0]
        print(f"➡️  Embedding {label.lower()}: {name}")
        embedder = embedder_class(model, name)
        embedder()
    print(f"✅ All {label.lower()}s embedded.\n")


def main():
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    embed_all_files(
        embedder_class=ArticleEmbedder,
        directory="data/raw/articles",
        model=model,
        label="Article"
    )

    embed_all_files(
        embedder_class=SubjectEmbedder,
        directory="data/raw/subjects",
        model=model,
        label="Subject"
    )

if __name__ == "__main__":
    main()
