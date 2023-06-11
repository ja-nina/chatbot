import streamlit as st
import torch
from streamlit_chat import message
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatBot:
    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.chat_history_ids = None

    def get_response(self, user_input: str) -> str:
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt")
        bot_input_ids = torch.cat(
            [self.chat_history_ids, new_user_input_ids], dim=-1) if self.chat_history_ids else new_user_input_ids

        chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8,
        )

        return self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


def init_streamlit():
    st.title("Gombrowicz Chat")
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "temp" not in st.session_state:
        st.session_state["temp"] = ""


def main():
    init_streamlit()

    chatbot = ChatBot("../output-small/")

    def callback():
        st.session_state.temp = st.session_state.input
        st.session_state.past.append(st.session_state.temp)
        st.session_state.generated.append(chatbot.get_response(st.session_state.temp))
        st.session_state.input = ""

    st.text_input("You:", "", key="input", on_change=callback)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


if __name__ == "__main__":
    main()
