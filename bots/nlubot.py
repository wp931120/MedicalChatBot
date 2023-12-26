from paddlenlp import Taskflow
from py2neo import Graph


class NluBot:
    def __init__(self):
        # 语义槽
        self.slot = {
            "intent": None,
            "intent_score": None,
            "entitys": {}
        }
        # 图数据库
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
        self.slotFiller = SlotFiller()
        self.sqlParser = SqlParser()

    def run(self, text):
        try:
            # 获取语义槽
            self.slot = self.slotFiller.get_slot(text)
            print(self.slot)
            # 将语义槽解析成SQL
            sql = self.sqlParser.parse(self.slot)
            # 执行SQL并获取数据
            result = self.graph.run(sql)
            if self.slot["intent"] == "疾病的症状":
                result = result.to_data_frame()["n.name"]
            if self.slot["intent"] == "疾病的防御措施":
                result = result.to_data_frame()["m.prevent"]
            if self.slot["intent"] == "疾病产生的原因":
                result = result.to_data_frame()["m.cause"]
            if self.slot["intent"] == "患病该吃什么药":
                result = result.to_data_frame()["n.name"]
            if self.slot["intent"] == "药物能治疗什么病":
                result = result.to_data_frame()["n.name"]
            # 将数据组装成答案并返回
            result = ",".join(result.to_list())
            return result
        except Exception as e:
            print(e)
            return "我不知道"


class SlotFiller:
    def __init__(self):
        self.slot = {
            "intent": None,
            "intent_score": None,
            "entitys": {}
        }
        self.entity_schema = ['药物', '症状', '疾病']
        self.intent_schema = ["疾病的防御措施", "疾病产生的原因", "患病该吃什么药", "疾病的症状", "药物能治疗什么病"]
        self.entity_extract = Taskflow('information_extraction', schema=self.entity_schema)
        self.intent_cls = Taskflow("zero_shot_text_classification", model="utc-base", schema=self.intent_schema)

    def get_slot(self, text):
        # 意图识别
        intent = self.intent_cls(text)
        # 词槽抽取
        entitys = self.entity_extract(text)
        if entitys is not None:
            for i in entitys[0].items():
                self.slot["entitys"][i[0]] = {i[1][0]["text"]: {i[1][0]["probability"]}}

        if intent is not None and len(intent[0]["predictions"]) > 0:
            if self.slot["intent"] != intent[0]["predictions"][0]["label"]:
                self.slot["intent"] = intent[0]["predictions"][0]["label"]
                self.slot["intent_score"] = intent[0]["predictions"][0]["score"]

        return self.slot


class SqlParser:
    # SQL解析类
    def __init__(self):
        pass

    def parse(self, slot):
        if slot["intent"] == "疾病的症状":
            if '疾病' in slot["entitys"].keys():
                entity = list(slot["entitys"]['疾病'].keys())[0]
                return "MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) where m.name = '{}' return m.name, n.name".format(entity)
        if slot["intent"] == "疾病的防御措施":
            if '疾病' in slot["entitys"].keys():
                entity = list(slot["entitys"]['疾病'].keys())[0]
                return "MATCH (m:Disease) where m.name = '{}' return m.name, m.prevent".format(entity)
        if slot["intent"] == "疾病产生的原因":
            if '疾病' in slot["entitys"].keys():
                entity = list(slot["entitys"]['疾病'].keys())[0]
                return "MATCH (m:Disease) where m.name = '{}' return m.name, m.cause".format(entity)
        if slot["intent"] == "患病该吃什么药":
            if '疾病' in slot["entitys"].keys():
                entity = list(slot["entitys"]['疾病'].keys())[0]
                return "MATCH (m:Disease)-[r:common_drug]->(n:Drug) where m.name = '{}' return m.name, n.name".format(entity)
        if slot["intent"] == "药物能治疗什么病":
            if '药物' in slot["entitys"].keys():
                entity = list(slot["entitys"]['药物'].keys())[0]
                return "MATCH (m:Disease)-[r:common_drug]->(n:Drug) where n.name = '{}' return m.name, r.name, n.name".format(entity)


if __name__ == '__main__':
    nluBot = NluBot()
    response = nluBot.run("高血压吃什么药")
    print(response)