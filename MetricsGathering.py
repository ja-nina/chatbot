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
        start = time.time()
        responses = []
        for text in self.data:
            responses.append(self.chatbot.get_response(text))
        end = time.time()
        return responses, (end - start)/len(responses)

    def get_response_length(self, responses: List[str]) -> float:
        return mean([len(response) for response in responses])

    def get_grammar_of_response(self, responses: List[str]) -> float:
        def _find_number_of_errors(text):
            matches = self.tool.check(text)
            return len(matches)

        return mean(list(map(_find_number_of_errors, responses)))

    def gather_metrics(self):
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
