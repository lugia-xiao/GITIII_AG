import torch
import torch.nn as nn

class ProportionalMasking_differentiable(nn.Module):
    def __init__(self, prob, temperature=0.1):
        super(ProportionalMasking_differentiable, self).__init__()
        self.prob = prob
        # Temperature controls the sharpness of the sigmoid function. Lower values make it sharper.
        self.temperature = temperature

    def forward(self, x):
        if not self.training:
            return x

        # Calculate the absolute values and their proportional importance
        proportional_importance = torch.abs(x) / torch.sum(torch.abs(x), dim=1, keepdim=True)
        proportional_importance = 1 / proportional_importance
        proportional_importance = proportional_importance / torch.sum(proportional_importance, dim=1, keepdim=True)
        # proportional_importance=torch.square(proportional_importance)*proportional_importance
        # proportional_importance=proportional_importance/torch.sum(proportional_importance,dim=1,keepdim=True)

        # Calculate the masking probability for each element
        mask_prob = self.prob * proportional_importance

        # Use a sigmoid function to create a soft mask
        random_tensor = torch.rand_like(x)
        soft_mask = torch.sigmoid((random_tensor - mask_prob) / self.temperature)

        # Apply the soft mask
        x_masked = x * soft_mask
        return x_masked

class ProportionalMasking(nn.Module):
    def __init__(self, prob):
        super(ProportionalMasking, self).__init__()
        # Ensure the probability is a PyTorch tensor so it can be used in calculations that involve other tensors.
        self.prob = prob

    def forward(self, x):
        if not self.training:
            # If the model is in evaluation mode, return the input tensor unchanged
            return x

        # Calculate the absolute values and their proportional importance
        proportional_importance = torch.abs(x) / torch.sum(torch.abs(x), dim=1, keepdim=True)
        proportional_importance=1/proportional_importance
        thresholds=proportional_importance/torch.sum(proportional_importance,dim=1,keepdim=True)

        #proportional_importance=torch.square(proportional_importance)*proportional_importance
        #proportional_importance=proportional_importance/torch.sum(proportional_importance,dim=1,keepdim=True)

        # Calculate the masking probability for each element
        mask_prob = self.prob * thresholds

        # Generate a random tensor from a uniform distribution and compare it with mask_prob
        # Elements are masked (set to zero) where the random values are less than the calculated mask_prob
        random_tensor = torch.rand_like(x)
        mask = (random_tensor < mask_prob).float()  # This creates a mask of 0s and 1s

        # Apply the mask
        x_masked = x * (1-mask)
        return x_masked

class ProportionalMasking_importance(nn.Module):
    def __init__(self, prob, resolution=2):
        super(ProportionalMasking_importance, self).__init__()
        # Ensure the probability is a PyTorch tensor so it can be used in calculations that involve other tensors.
        self.prob = prob
        self.resolution=resolution

    def forward(self, x):
        if not self.training:
            # If the model is in evaluation mode, return the input tensor unchanged
            return x

        # Calculate the absolute values and their proportional importance
        y_pred=torch.sum(x, dim=1, keepdim=True)
        proportional_importance = torch.abs(1-(y_pred-x) / y_pred)
        thresholds=1/proportional_importance#torch.nn.functional.softmax(-proportional_importance*self.resolution,dim=1)
        thresholds=thresholds/torch.sum(thresholds,dim=1,keepdim=True)

        # Calculate the masking probability for each element
        mask_prob = self.prob * thresholds

        # Generate a random tensor from a uniform distribution and compare it with mask_prob
        # Elements are masked (set to zero) where the random values are less than the calculated mask_prob
        random_tensor = torch.rand_like(x)
        mask = (random_tensor < mask_prob).float()  # This creates a mask of 0s and 1s

        # Apply the mask
        x_masked = x * (1-mask)
        return x_masked


class ProportionalMasking_cumsum(nn.Module):
    def __init__(self,temperature=2):
        super(ProportionalMasking_cumsum, self).__init__()
        self.temperature=temperature

    def forward(self, x):
        if not self.training:
            # If the model is in evaluation mode, just return x
            return x

        # Calculate the proportional importance
        proportional_importance = torch.abs(x) / torch.sum(torch.abs(x), dim=1, keepdim=True)
        proportional_importance =  torch.exp(proportional_importance*self.temperature)

        # Calculate thresholds
        sum_proportional_importance = torch.sum(proportional_importance, dim=1, keepdim=True)
        thresholds = proportional_importance / sum_proportional_importance

        # Randomly select thresholds according to the calculated weights
        cumulative_thresholds = thresholds.cumsum(dim=1)+1e-3
        random_values = torch.rand(cumulative_thresholds.shape[0], cumulative_thresholds.shape[2],
                                   device=x.device).unsqueeze(1)

        #print(random_values,cumulative_thresholds,cumulative_thresholds < random_values)
        # Determine the indices where the random values fall into the cumulative thresholds
        chosen_indices = (cumulative_thresholds <= random_values).sum(dim=1, keepdim=True)

        #print(torch.gather(torch.abs(x), 1, chosen_indices).shape)
        # Create a mask where values are zeroed out below the chosen threshold
        mask = torch.abs(x) >= torch.gather(torch.abs(x), 1, chosen_indices)

        # Apply mask
        return x * mask

if __name__=="__main__":
    # Example usage:
    prob = 0.5
    B, N, C = 1,7,2#512, 49, 140  # Example dimensions for the input tensor
    x = torch.exp(torch.randn(B, N, C))  # Random tensor simulating input

    print(x)
    masking_module = ProportionalMasking_cumsum()#(prob)
    output = masking_module(x)
    print(output)
