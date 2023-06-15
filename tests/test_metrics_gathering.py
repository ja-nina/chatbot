from unittest.mock import MagicMock

from chatbot.metrics_gathering import MetricsGathering


def test_get_responses():
    # given
    chatbot = MagicMock()
    data = ['text_1', 'text_2']
    tool = MagicMock()
    classifier = MagicMock()
    chatbot.get_response = lambda x: "response"

    metrics_gathering = MetricsGathering(
        chatbot, data, tool, classifier)

    # when
    responses, time = metrics_gathering.gather_responses_and_time()

    # then
    assert responses == ["response", "response"]
    assert type(time) == float


def test_get_response_length():
    # given
    chatbot = MagicMock()
    data = ['text_1', 'text_2']
    tool = MagicMock()
    classifier = MagicMock()

    metrics_gathering = MetricsGathering(
        chatbot, data, tool, classifier)

    # when
    mean_len = metrics_gathering.get_response_length(["a", "abc"])

    # then
    assert mean_len == 2


def test_get_grammar():
    # given
    chatbot = MagicMock()
    data = ['text_1', 'text_2']
    tool = MagicMock()
    tool.check = lambda x: {
        "response1": [1],
        "response2": [1, 2],
    }[x]
    classifier = MagicMock()

    metrics_gathering = MetricsGathering(
        chatbot, data, tool, classifier)

    # when
    error_mean = metrics_gathering.get_grammar_of_response(
        ["response1", "response2"])

    # then
    assert error_mean == 1.5
