from datetime import datetime
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


class CosineSimilarityEvaluator:
    def __init__(self, subject, article):
        subject_name = os.path.splitext(subject)[0]
        article_name = os.path.splitext(article)[0]

        self.embedded_subject = torch.load(f"data\\embeddings\\subjects\\{subject_name} embedding.pt")
        self.subject = subject_name
        self.article = article_name

        with open(f"data\\embeddings\\articles\\{article_name} embedding.json", "r", encoding="utf-8") as f:
            self.article_data = json.load(f)

        self.cosine_scores = None

    def compute_cosine_scores(self):
        embeddings = [
            torch.tensor(data["embedding"], dtype=torch.float32)
            for data in self.article_data.values()
        ]
        embedded_article = torch.stack(embeddings)

        if self.embedded_subject.dim() == 1:
            embedded_subject = self.embedded_subject.unsqueeze(0).repeat(embedded_article.size(0), 1)
        else:
            embedded_subject = self.embedded_subject

        self.cosine_scores = F.cosine_similarity(embedded_subject, embedded_article, dim=1)

    def save_plot(self, fig, graph_title):
        dir_path = "results"
        os.makedirs(dir_path, exist_ok=True)
        save_path = os.path.join(dir_path, f"{graph_title}.png")
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    def plot_similarity_over_time(self):
        if self.cosine_scores is None:
            self.compute_cosine_scores()

        date_strs = [v["date"] for v in self.article_data.values()]
        dates = [datetime.strptime(ds, "%b %Y") for ds in date_strs]

        sorted_pairs = sorted(zip(dates, self.cosine_scores.cpu().numpy()), key=lambda x: x[0])
        sorted_dates, sorted_scores = zip(*sorted_pairs)

        # Define pastel colors based on article name
        color_map = {
            "Libération": "#e67676",  # pastel red
            "Le Monde": "#80d880",    # pastel green
            "Le Figaro": "#3f84df"    # pastel blue
        }
        dot_color = color_map.get(self.article, "gray")  # default to gray if not matched

        article_and_subject = f"{self.article} - {self.subject}"
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sorted_dates, sorted_scores, marker='o', linestyle='None', color=dot_color)  # colored points
        ax.set_title(f"Cosine Similarity Over Time: {article_and_subject}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cosine Similarity")
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        self.save_plot(fig, article_and_subject)
        plt.close(fig)
        plt.show()

    def plot_smoothings(self, window_size=10, sigma=2):
        if self.cosine_scores is None:
            self.compute_cosine_scores()

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
            self.compute_cosine_scores()

        scores = self.cosine_scores.cpu().numpy()
        top_indices = np.argsort(scores)[-n:][::-1]
        top_scores = scores[top_indices]
        return list(zip(top_indices, top_scores))
