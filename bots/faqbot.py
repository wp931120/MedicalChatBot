from sentence_transformers import SentenceTransformer
import lancedb


class FAQBot:

    def __init__(self):
        # 向量编码模型
        self.encode_model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
        # 向量检索数据库连接
        self.db = lancedb.connect("D:\Pycharm\MedicalChatBot/data/sample-lancedb")
        self.table = self.db.open_table("medical")

    def run(self, query):
        # 将query 用向量模型编码
        embeddings_query = self.encode_model.encode(query, normalize_embeddings=True)
        # 向量数据库，进行QQ匹配，搜索QA库中最相似的Q，并返回其对应的A
        data_embed = self.table.search(embeddings_query).limit(1)
        return data_embed.to_pandas()["answer"][0]


if __name__ == '__main__':
    faq = FAQBot()
    response = faq.run("高血压怎么办")
    print(response)