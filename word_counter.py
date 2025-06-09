import re
from collections import Counter

def load_keywords(file_path):
    """
    Load keywords from a file, one per line.

    Parameters:
        file_path (str): Path to the file containing keywords.

    Returns:
        list: A list of cleaned keywords.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def count_keywords_in_file(text_file_path, keywords):
    """
    Count how many times each keyword appears in a given text file.

    Parameters:
        text_file_path (str): Path to the text file to analyze.
        keywords (list): List of keywords to count.

    Returns:
        dict: A dictionary mapping keywords to their counts.
    """
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()

    counts = Counter()
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, text)
        counts[keyword] = len(matches)

    return dict(counts)

def sum_keyword_counts(counts):
    """
    Sum the values of keyword counts.

    Parameters:
        counts (dict): Dictionary of keyword counts.

    Returns:
        int: Total keyword occurrences.
    """
    return sum(counts.values())

def count_total_words_in_file(file_path):
    """
    Count the total number of words in the specified text file.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        int: Total number of words in the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Use regex to split by word characters
    words = re.findall(r'\b\w+\b', text)
    return len(words)


if __name__ == "__main__":
    keyword_file = 'C:\\Users\\maxba\\Desktop\\embedding-cosine-similarity-analysis\\data\\phrases_list.txt'   # File with one keyword or phrase per line
    text_file = 'C:\\Users\\maxba\\Desktop\\embedding-cosine-similarity-analysis\\data\\raw\\articles\\Libération.txt'         # Text file to analyze

    keywords = load_keywords(keyword_file)
    keyword_counts = count_keywords_in_file(text_file, keywords)
    total = sum_keyword_counts(keyword_counts)

    print("Individual counts:")
    for word, count in keyword_counts.items():
        print(f"{word}: {count}")

    print(f"\nTotal keyword occurrences: {total}")

    print(f"Total words in document: {count_total_words_in_file(text_file)}")

