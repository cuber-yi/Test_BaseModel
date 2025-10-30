import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, enc_in, pred_len, seq_len, hidden_size=128, num_layers=2):
        super(RNN, self).__init__()
        self.pred_len = pred_len

        # --- 联邦共享部分 ('trend') ---
        self.rnn = nn.RNN(
            input_size=enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.global_head = nn.Linear(hidden_size, pred_len * 2)

        # --- 个性化部分 ('personal') ---
        self.personal_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(pred_len * 2, pred_len)
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_time_step_out = rnn_out[:, -1, :]
        out = self.global_head(last_time_step_out)
        out = self.personal_head(out)
        return out
