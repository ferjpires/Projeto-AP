import random

def build_random_search(dictionary, n_iter=20, random_state=42):
    random.seed(random_state)
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    for _ in range(n_iter):
        instance = [random.choice(v) for v in values]
        yield dict(zip(keys, instance))
