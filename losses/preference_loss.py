import math
import torch.nn as nn
import torch

class PreferenceLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PreferenceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prob_pref_traj_1: torch.Tensor, prob_pref_traj_2: torch.Tensor, mu1: torch.Tensor, mu2: torch.Tensor):
        """
        Compute the custom loss based on human preferences.

        Parameters:
            prob_pref_traj_1: torch.Tensor
                Estimated probability that the evaluator prefers trajectory (s, a) over (s', a')
            prob_pref_traj_2: torch.Tensor
                Estimated probability that the evaluator prefers trajectory (s', a') over (s, a)
            real_preference: torch.Tensor
                Empirical frequency of preference (s', a') over (s, a) [0, 0.5, 1]

        Returns:
            loss: torch.Tensor
                The custom loss value.
        """
        # Preventing very small values close to 0 to prevent NaNs
        prob_pref_traj_1 = torch.clamp(prob_pref_traj_1, self.epsilon, 1.0 - self.epsilon)
        prob_pref_traj_2 = torch.clamp(prob_pref_traj_2, self.epsilon, 1.0 - self.epsilon)

        loss = -mu1 * torch.log(prob_pref_traj_1) - mu2 * torch.log(prob_pref_traj_2)

        return loss
