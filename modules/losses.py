import torch
from .SO3lib import produce_sine_grid


class S2RMSELoss(torch.nn.modules.loss._Loss):
    """
    Compute the weighted RMSE loss on the sphere S2, with sine-weighting.
    If normalization is True, the RMSE is normalized by the L2 norm of the input.
    """
    def __init__(self, n_beta = 179, normalization = False) -> None:
        super().__init__()
        self.n_beta = n_beta
        self.normalization = normalization

    def forward(self, input, target):
        w = produce_sine_grid(input.device, self.n_beta)

        diff2 = torch.abs(input - target) ** 2 
        weighted_mse = (diff2 * w).sum(dim=(-1, -2)) / w.sum()
        rmse = torch.sqrt(weighted_mse)  # [batch, features] or [batch,]
        
        if self.normalization:
            input2 = torch.abs(input) ** 2
            norm2_input = (input2 * w).sum(dim=(-1, -2)) / w.sum()
            norm_input = torch.sqrt(norm2_input)
            rmse = rmse / (norm_input + 1e-12)

        rmse_mean = rmse.mean(dim=0)  # [features,] or scalar
        return rmse_mean


class S2MSELoss(torch.nn.modules.loss._Loss):
    """
    Mean Squared Error on the sphere S2, with sine-weighting.
    If normalization is True, the MSE is normalized by the L2 norm of the input.
    """
    def __init__(self, n_beta = 179, normalization = False) -> None:
        super().__init__()
        self.n_beta = n_beta
        self.normalization = normalization

    def forward(self, input, target):
        w = produce_sine_grid(input.device, self.n_beta)

        diff2 = torch.abs(input - target) ** 2 
        weighted_mse = (diff2 * w).sum(dim=(-1, -2)) / w.sum()

        if self.normalization:
            input2 = torch.abs(input) ** 2
            norm2_input = (input2 * w).sum(dim=(-1, -2)) / w.sum()
            weighted_mse = weighted_mse / (norm2_input + 1e-12)

        mse = weighted_mse.mean() # [features,] or scalar
        return mse
        