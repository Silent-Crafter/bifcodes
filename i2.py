import nltk
from collections import defaultdict
from typing import List, Dict

class InvertedIndex:
    def __init__(self):
        self.terms: Dict[str, List[int]] = defaultdict(list)
        self.term_freq: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def build_index(self, docs: List[str]) -> None:
        for doc_id, doc in enumerate(docs):
            words = nltk.tokenize.word_tokenize(doc)
            for word in words:
                self.terms[word].append(doc_id)
                self.term_freq[word][doc_id] += 1

    def search(self, query: str) -> List[int]:
        terms = nltk.tokenize.word_tokenize(query)
        result_docs = set(self.terms[terms[0]])
        for term in terms:
            if term in self.terms:
                result_docs = result_docs.intersection(self.terms[term])

        return sorted(result_docs)


if __name__ == "__main__":
    inv_index = InvertedIndex()
    documents = [
        "The cat sat on the mat. The mat is good.",
        "The dog barked. The cat ran away.",
        "Car is the best for the cat."
    ]


    inv_index.build_index(documents)
    print(f"Documents {inv_index.search('cat the')}")

