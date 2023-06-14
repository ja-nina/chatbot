import torch
import pandas as pd
import language_check
import logging
import time
from statistics import mean
from typing import Tuple, List
from frontend.chatbot import ChatBot


class MetricsGathering:
    def __init__(self, chatbot: ChatBot, data_location: str):
        logging.basicConfig(level=logging.INFO)
        self.chatbot = chatbot
        self.data = pd.read_csv(data_location).head(100)['Text']
        self.tool = language_check.LanguageTool('en-US')
        self.logger = logging.getLogger("metrics_logger")

    def gather_responses_and_time(self) -> Tuple[List[str], float]:
        """Get Chatbot Responses and time it takes for generating

        Get Chatbor responses generated by feeding the Chatbot texts
        for generic dataset, function also measures how much time
        it took to generate them.


        Returns:
            List of responses and mean response time
        """
        start = time.time()
        responses = []
        for text in self.data:
            responses.append(self.chatbot.get_response(text))
        end = time.time()
        return responses, (end - start)/len(responses)

    def get_response_length(self, responses: List[str]) -> float:
        """Get mean length

        Simple function to return mean length of text.

        Args:
            responses (List[str]) : List of responses

        Returns:
            Mean length of item in ``responses``.

        """
        return mean([len(response) for response in responses])

    def get_grammar_of_response(self, responses: List[str]) -> float:
        """Get correctness of texts

        Simple function to return corectness of text,
        by check on rule-based engine.

        Args:
            responses (List[str]) : List of responses

        Returns:
            Mean mistakes ``responses``.

        """
        def _find_number_of_errors(text):
            matches = self.tool.check(text)
            return len(matches)

        return mean(list(map(_find_number_of_errors, responses)))

    def gather_metrics(self):
        """Get statistics of metrics: length, time and correctness.

        Simple function to return corectness of text,
        by check on rule-based engine.

        """
        responses, mean_time = self.gather_responses_and_time()
        self.logger.info(f"Length of respose : \
                         {self.get_response_length(responses)}")
        self.logger.info(f"Get mistakes rate : \
                         {self.get_grammar_of_response(responses)}")
        self.logger.info(f"Get mean time rate: {mean_time}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chatbot = ChatBot("./output-small/", device)
    metrics = MetricsGathering(chatbot, 'data_metrics\\dialogs.csv')
    metrics.gather_metrics()


main()
