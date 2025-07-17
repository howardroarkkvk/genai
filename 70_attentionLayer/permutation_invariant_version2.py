import torch
import torch.nn as nn

class PermutationInvariantNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Step 1: Apply a shared function to each element
        self.element_fn = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU()
        )
        # Step 2: Aggregate (we'll use sum here)
        # Step 3: Process the pooled result
        self.aggregate_fn = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        # x: (batch, set_size, 1)
        # Apply the same function to each element
        x = self.element_fn(x)  # shape: (batch, set_size, 4)
        print('after elementtal ouput',x,x.shape)
        x = x.sum(dim=1)        # sum over set dimension => shape: (batch, 4)
        print('after sum  ouput',x,x.shape)
        out = self.aggregate_fn(x)  # final output
        print('after aggregate ouput',x,x.shape)
        return out
if __name__=='__main__':
    model = PermutationInvariantNN()

    # Create two sets with same elements in different order
    set1 = torch.tensor([[1.0], [2.0], [3.0]])  # shape: (3, 1)
    set2 = torch.tensor([[3.0], [1.0], [2.0]])  # permuted

    # Add batch dimension
    set1 = set1.unsqueeze(0)  # shape: (1, 3, 1)
    set2 = set2.unsqueeze(0)

    # Get outputs
    print('set 1 is :',set1, set1.shape)
    print('set 2 is :',set2, set2.shape)
    out1 = model(set1)
    out2 = model(set2)

    print("Output 1:", out1.item())
    print("Output 2:", out2.item())
