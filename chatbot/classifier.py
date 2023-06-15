import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


class Classifier:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def get_input_tokens(self, user_input: str) -> torch.Tensor:
        """Get tokens from model

        Pass input to tokenizer to produce tokens.
        Moves the output to device.

        Args:
            user_input (str) : input text

        Returns:
            torch Tensor with tokens
        """
        return self.tokenizer(
            user_input, return_tensors="pt").to(self.device)

    def get_prediction(self, user_input: str) -> int:
        """Get prediction from model

        Pass input to tokenizer to produce tokens.
        Moves the output to device.
        Pass the tokens to the model along with context.

        Args:
            user_input (str) : input text

        Returns:
            predictions from model
        """
        inputs = self.get_input_tokens(user_input)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)

        return int(predicted_class.item())
