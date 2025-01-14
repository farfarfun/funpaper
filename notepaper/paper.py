__all__ = ["get_paper_list"]
import os


def get_paper_list():
    paper_list = [
        {
            "name": "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction",
            "url": "https://arxiv.org/pdf/1703.04247.pdf",
            "year": 2017,
        },
        {"url": "https://arxiv.org/pdf/1703.04247.pdf"},
    ]

    return paper_list


papers = get_paper_list()

print(os.listdir("../"))

doc = open("../README.md", "r").read()
print(doc)
