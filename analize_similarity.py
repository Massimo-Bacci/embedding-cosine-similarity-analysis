import json
from cosine_similarity_evaluator import CosineSimilarityEvaluator

def load_config(path='config\\config.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    config = load_config()

    subject = config.get("SUBJECT")
    article = config.get("ARTICLE")

    evaluator = CosineSimilarityEvaluator(subject, article)
    print(evaluator.cosine_scores)
    evaluator.plot_smoothings()

if __name__ == "__main__":
    main()