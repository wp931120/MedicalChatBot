from sentence_transformers import SentenceTransformer
import lancedb


class FAQBot:

    def __init__(self):
        self.encode_model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
        self.db = lancedb.connect("D:\Pycharm\MedicalChatBot/data/sample-lancedb")
        self.table = self.db.open_table("medical")

    def run(self, query):
        embeddings_query = self.encode_model.encode(query, normalize_embeddings=True)
        data_embed = self.table.search(embeddings_query).limit(1)
        return data_embed.to_pandas()["answer"][0]


if __name__ == '__main__':
    faq = FAQBot()
    response = faq.run("高血压怎么办")
    print(response)