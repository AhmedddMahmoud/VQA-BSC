from __future__ import annotations

import torch
import torch.nn as nn


class QuestionEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        rnn_hidden = hidden_dim // 2 if bidirectional else hidden_dim

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=rnn_hidden,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "bilstm" or self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=rnn_hidden,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

    def forward(
        self,
        question_ids: torch.Tensor,
        question_lengths: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | dict:
        embedded = self.embedding(question_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths=question_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.rnn(packed)
        _ = packed_output

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.bidirectional:
            forward_last = hidden[-2]
            backward_last = hidden[-1]
            question_vector = torch.cat([forward_last, backward_last], dim=1)
        else:
            question_vector = hidden[-1]

        if return_intermediate:
            return {
                "embedded": embedded,
                "question_vector": question_vector,
            }

        return question_vector
