import streamlit as st
import torch
from streamlit_chat import message

from src.chatbot import ChatBot


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chatbot = ChatBot("../output-small/", device)

    def callback():
        st.session_state.temp = st.session_state.input
        st.session_state.past.append(st.session_state.temp)
        st.session_state.generated.append(chatbot.get_response(
            st.session_state.temp))
        st.session_state.input = ""

    st.text_input("You:", "", key="input", on_change=callback)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i],
                    is_user=True, key=str(i) + "_user")


if __name__ == "__main__":
    main()
