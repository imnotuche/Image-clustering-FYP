import torch
import torch.optim as optim
from dec_model import DEC

def target_distribution(q):
    """
    The 'P' distribution in DEC. 
    It takes the soft assignments (q) and squares them to make the 
    highest probabilities even stronger.
    """
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_dec(dataloader, n_clusters=10, epochs=50):
    # 1. Initialize our Brain
    model = DEC(n_clusters=n_clusters)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')

    model.train()
    
    for epoch in range(epochs):
        for images, labels in dataloader:
            # Clear previous gradients
            optimizer.zero_grad()
            
            # Forward pass: Get soft assignments (q)
            q = model(images)
            
            # Calculate the target distribution (p)
            # This is the "Gold Standard" the model tries to reach
            p = target_distribution(q).detach()
            
            # Calculate Loss (KL Divergence)
            # This measures the distance between 'q' and 'p'
            loss = criterion(q.log(), p)
            
            # Backward pass: Update the weights and cluster centers
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model