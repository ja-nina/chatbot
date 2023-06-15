import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatBot:
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.chat_history_ids = None

    def get_input_tokens(self, user_input: str) -> torch.Tensor:
        """Get tokens from model

        Pass input to tokenizer to produce tokens.
        Moves the output to device.

        Args:
            user_input (str) : input text

        Returns:
            torch Tensor with tokens
        """
        return self.tokenizer.encode(
            user_input + self.tokenizer.eos_token, return_tensors="pt"
        ).to(self.device)

    def get_response(self, user_input: str) -> str:
        """Get response from model

        Pass input to tokenizer to produce tokens.
        Moves the output to device.
        Pass the tokens to the model along with context.

        Args:
            user_input (str) : input text

        Returns:
            text response from model
        """
        new_user_input_ids = self.get_input_tokens(user_input)
        bot_input_ids = (
            torch.cat([torch.Tensor(self.chat_history_ids),
                       torch.Tensor(new_user_input_ids)], dim=-1)
            if self.chat_history_ids else new_user_input_ids
        )

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

        return self.tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True,
        )
