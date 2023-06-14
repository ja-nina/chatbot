import torch

from frontend.chatbot import ChatBot


def test_get_input_tokens_from_chatbot():
    # given
    chatbot = ChatBot("./output-small/", torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))

    # when
    response = chatbot.get_response("custom response")

    # then
    assert type(response) == str
    assert len(response) > 0
