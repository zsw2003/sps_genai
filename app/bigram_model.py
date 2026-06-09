import random
from collections import defaultdict


class BigramModel:
    def __init__(self, corpus):
        self.model = defaultdict(list)
        self.build_model(corpus)

    def build_model(self, corpus):
        for sentence in corpus:
            words = sentence.lower().replace(".", "").replace(",", "").split()

            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i + 1]
                self.model[current_word].append(next_word)

    def generate_text(self, start_word, length):
        start_word = start_word.lower()
        words = [start_word]
        current_word = start_word

        for _ in range(length - 1):
            if current_word not in self.model:
                break

            next_word = random.choice(self.model[current_word])
            words.append(next_word)
            current_word = next_word

        return " ".join(words)