from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

class CosineSimilarityEvaluator:
    def __init__(self, subject, article):

        embedded_subject = torch.load(f"data\\embeddings\\subjects\\{subject} embedding.pt")
        embedded_article = torch.load(f"data\\embeddings\\articles\\{article} embedding.pt")

        self.cosine_scores = F.cosine_similarity(embedded_subject, embedded_article, dim=1)

    def plot_smoothings(self, window_size=10, sigma=2):
        scores = self.cosine_scores.cpu().numpy()

        def moving_average(x, w):
            return np.convolve(x, np.ones(w)/w, mode='valid')

        ma_scores = moving_average(scores, window_size)
        gauss_scores = gaussian_filter1d(scores, sigma=sigma)

        plt.figure(figsize=(12, 6))
        plt.plot(scores, label="Raw Cosine Scores", alpha=0.5)
        plt.plot(range(window_size - 1, len(scores)), ma_scores, label=f"Moving Average (window={window_size})", linewidth=2)
        plt.plot(gauss_scores, label=f"Gaussian Smoothing (sigma={sigma})", linewidth=2)
        plt.title("Cosine Similarity Scores with Smoothing")
        plt.xlabel("Paragraph Index")
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_top_n_scores(self, n=5):

        scores = self.cosine_scores.cpu().numpy()
        top_indices = np.argsort(scores)[-n:][::-1]
        top_scores = scores[top_indices]

        return list(zip(top_indices, top_scores))

if __name__ == "__main__":
    evaluator = CosineSimilarityEvaluator(
        "Cadrage par appartenance nationale-racialisée",
        "Europresse Le Figaro"
    )
    print(evaluator.cosine_scores)
    evaluator.plot_smoothings()
