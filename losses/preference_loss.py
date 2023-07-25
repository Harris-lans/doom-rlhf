import math
import torch.nn as nn

class PreferenceLoss(nn.Module):
    def __init__(self):
        super(PreferenceLoss, self).__init__()

    def forward(self, prob_pref_traj_1, prob_pref_traj_2, real_preference):
        """
        Compute the custom loss based on human preferences.

        Parameters:
            prob_pref_traj_1: float
                Estimated probability that the evaluator prefers trajectory (s, a) over (s', a')
            prob_pref_traj_2: float
                Estimated probability that the evaluator prefers trajectory (s', a') over (s, a)
            real_preference: float
                Empirical frequency of preference (s', a') over (s, a) [0, 0.5, 1]

        Returns:
            loss: torch.Tensor
                The custom loss value.
        """
        assert real_preference in (0, 0.5, 1), "Invalid preference value"

        if real_preference == 0:
            mu1 = 1
            mu2 = 0
        elif real_preference == 0.5:
            mu1 = 0.5
            mu2 = 0.5
        else:
            mu1 = 0
            mu2 = 1

        loss = -mu1 * math.log(prob_pref_traj_1) - mu2 * math.log(prob_pref_traj_2)

        return loss
    
    def backward(self, grad_output):
        """
        Compute the gradient of the custom loss with respect to the input.

        Parameters:
            grad_output: torch.Tensor
                Gradient of the loss with respect to the output of the forward pass.

        Returns:
            gradient: torch.Tensor
                Gradient of the loss with respect to the input.
        """
        # Since the loss is a scalar value, we propagate the gradient directly
        return grad_output
