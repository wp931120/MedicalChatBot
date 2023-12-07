import gradio as gr
from bots.faqbot import FAQBot
from bots.llmbot import LLMBot
from bots.nlubot import NluBot

faqbot = FAQBot()
nlubot = NluBot()  # 创建NLU机器人实例
llmbot = LLMBot()  # 创建LLM机器人实例

custom_css = """
.tab_item {
    font-size: 20px; /* 调整选项卡标题的字体大小 */
}
"""


def faq_respond(message, chat_history):
    bot_message = faqbot.run(message)
    chat_history.append((message, bot_message))
    return "", chat_history


def nlu_respond(message, chat_history):
    bot_message = nlubot.run(message)
    chat_history.append((message, bot_message))
    return "", chat_history


def llm_respond(message, chat_history):
    bot_message = llmbot.run(message)
    chat_history.append((message, bot_message))
    return "", chat_history


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("<center>")  # 开始居中
        gr.Image("./Logov2.png", scale=0.3)
        gr.Markdown("</center>")  # 结束居中
    gr.Markdown("""<center><font size=8>Medical Bot</center>""")

    with gr.Tabs() as tabs:
        with gr.TabItem("FAQ"):
            faq_chat = gr.Chatbot()
            faq_msg = gr.Textbox(placeholder="Type your message here...")
            faq_submit = gr.Button("Submit")
            faq_clear = gr.ClearButton([faq_msg, faq_chat])
            faq_submit.click(faq_respond, [faq_msg, faq_chat], [faq_msg, faq_chat])

        with gr.TabItem("NLU"):
            nlu_chat = gr.Chatbot()
            nlu_msg = gr.Textbox(placeholder="Type your message here...")
            nlu_submit = gr.Button("Submit")
            nlu_clear = gr.ClearButton([nlu_msg, nlu_chat])
            nlu_submit.click(nlu_respond, [nlu_msg, nlu_chat], [nlu_msg, nlu_chat])

        with gr.TabItem("LLM"):
            llm_chat = gr.Chatbot()
            llm_msg = gr.Textbox(placeholder="Type your message here...")
            llm_submit = gr.Button("Submit")
            llm_clear = gr.ClearButton([llm_msg, llm_chat])
            llm_submit.click(llm_respond, [llm_msg, llm_chat], [llm_msg, llm_chat])


if __name__ == "__main__":
    demo.launch()