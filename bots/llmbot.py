import lancedb
import requests
import json
from sentence_transformers import SentenceTransformer


class LLMBot:

    def __init__(self):
        self.db = lancedb.connect("D:\Pycharm\MedicalChatBot/data/sample-lancedb")
        self.table = self.db.open_table("medical")
        self.model = SentenceTransformer('BAAI/bge-base-zh-v1.5')

    # RAG部分
    def search(self, query):
        # 将query 用向量模型编码
        embeddings_query = self.model.encode(query, normalize_embeddings=True)
        # 向量数据库，进行QQ匹配，搜索QA库中最相似的Q，并返回其对应的A
        results = self.table.search(embeddings_query).limit(3).to_pandas()
        results["q_a"] = results["query"] + ":" + results["answer"]
        document = "\n".join(results["q_a"].to_list())
        prompt = """
        请根据参考资料回答问题
        要求：
        1.不要跳出参考资料的范围去回答问题。
        2.如果参考资料的内容无法回答问题,则告知无法回答
        3.回答尽量简洁明了
        
        参考资料：
        {}
        
        问题：
        {}
        
        请给出你的回答：
        """.format(document, query)
        return prompt

    def run(self, query):
        # 语义向量检索
        prompt = self.search(query)
        # LLM生成
        result = self.llm(prompt)
        return result

    @staticmethod
    def llm(prompt):
        url = "https://api.baichuan-ai.com/v1/chat/completions"
        api_key = "5659bafa6a3ec2ff4af0ab3ed0038c9e"

        data = {
            "model": "Baichuan2",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }

        json_data = json.dumps(data)

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key
        }
        response = requests.post(url, data=json_data, headers=headers, timeout=60)
        if response.status_code == 200:
            print("请求成功！")
            return eval(response.text)["choices"][0]["message"]["content"]
        else:
            print("请求失败，状态码:", response.status_code)
            print("请求失败，body:", response.text)
            print("请求失败，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))


if __name__ == '__main__':
    llmBot = LLMBot()
    response = llmBot.run("高血压吃什么药")
    print(response)