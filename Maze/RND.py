import torch
import torch.nn as nn

class RND(nn.Module):
    def __init__(self, INPUT_DIM, OUTPUT_DIM):
        super(RND, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, OUTPUT_DIM)
        ).to(self.device)

        self.target = nn.Sequential(
            nn.Linear(INPUT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, OUTPUT_DIM)
        ).to(self.device)

        self.LR = 0.001
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()



        self.normalize_min_error = float("inf")
        self.normalize_max_error = float("-inf")

    def forward(self, state):
        predict = self.predictor(state)
        target = self.target(state)
        return predict, target

    def update_parameters(self, predict, target):
        loss = self.loss_func(predict, target)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.predictor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def normalize_error(self, predict_error):
        if predict_error < self.normalize_min_error:
            self.normalize_min_error = predict_error
        if predict_error > self.normalize_max_error:
            self.normalize_max_error = predict_error
        if self.normalize_min_error == self.normalize_max_error:
            self.normalize_min_error = self.normalize_max_error - 1E-3
        normalize_val = (predict_error - self.normalize_min_error) / (
                self.normalize_max_error - self.normalize_min_error)
        normalize_val = normalize_val * 0.1 + 0.1

        return normalize_val