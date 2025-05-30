from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

import json
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

class CosineSimilarityEvaluator:
    def __init__(self, subject, article):
        self.embedded_subject = torch.load(f"data\\embeddings\\subjects\\{subject} embedding.pt")

        with open(f"data\\embeddings\\articles\\{article} embedding.json", "r") as f:
            self.article_data = json.load(f)

        self.cosine_scores = None

    def compute_cosine_scores(self):
        embedded_article = torch.tensor(list(self.article_data.values()), dtype=torch.float)

        if self.embedded_subject.dim() == 1:
            embedded_subject = self.embedded_subject.unsqueeze(0).repeat(embedded_article.shape[0], 1)
        else:
            embedded_subject = self.embedded_subject

        self.cosine_scores = F.cosine_similarity(embedded_subject, embedded_article, dim=1)

    from datetime import datetime

    def plot_similarity_over_time(self):
        if not self.article_data:
            raise ValueError("No article data loaded.")

        # Parse dates and sort them
        date_embedding_pairs = [
            (datetime.strptime(k, "%b %Y"), v) for k, v in self.article_data.items()
        ]
        date_embedding_pairs.sort(key=lambda x: x[0])

        dates, embeddings = zip(*date_embedding_pairs)
        embedded_article = torch.tensor(embeddings, dtype=torch.float)

        # Repeat subject embedding if needed
        if self.embedded_subject.dim() == 1:
            embedded_subject = self.embedded_subject.unsqueeze(0).repeat(embedded_article.shape[0], 1)
        else:
            embedded_subject = self.embedded_subject

        # Compute cosine similarity
        cosine_scores = F.cosine_similarity(embedded_subject, embedded_article, dim=1).cpu().numpy()

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(dates, cosine_scores, marker='o', linestyle='-', linewidth=2)
        plt.title("Cosine Similarity Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cosine Similarity")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_smoothings(self, window_size=10, sigma=2):
        if self.cosine_scores is None:
            raise ValueError("Call compute_cosine_scores() before plotting.")

        scores = self.cosine_scores.cpu().numpy()

        def moving_average(x, w):
            return np.convolve(x, np.ones(w) / w, mode='valid')

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
        if self.cosine_scores is None:
            raise ValueError("Call compute_cosine_scores() before getting top scores.")

        scores = self.cosine_scores.cpu().numpy()
        top_indices = np.argsort(scores)[-n:][::-1]
        top_scores = scores[top_indices]
        return list(zip(top_indices, top_scores))
