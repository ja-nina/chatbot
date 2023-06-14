import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


class Classifier:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)

    def get_input_tokens(self, user_input: str) -> torch.Tensor:
        return self.tokenizer.encode(
            user_input + self.tokenizer.eos_token, return_tensors="pt"
            ).to(self.device)

    def get_response(self, user_input: str) -> int:
        new_user_input_ids = self.get_input_tokens(user_input)
        with torch.no_grad():
            outputs = self.model(**new_user_input_ids)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)

        return predicted_class.item()
