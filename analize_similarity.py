import os
from cosine_similarity_evaluator import CosineSimilarityEvaluator

def get_txt_filenames(directory):
    return [f for f in os.listdir(directory) if f.endswith('.txt')]

def main():
    subjects_dir = 'data/raw/subjects'
    articles_dir = 'data/raw/articles'

    subjects = get_txt_filenames(subjects_dir)
    articles = get_txt_filenames(articles_dir)

    for subject in subjects:
        for article in articles:
            evaluator = CosineSimilarityEvaluator(subject, article)
            evaluator.compute_cosine_scores()
            evaluator.plot_similarity_over_time()

if __name__ == "__main__":
    main()
