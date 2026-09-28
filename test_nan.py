import torch
from src.crato.experiments.train_crato import CRATOModel

def test_nans():
    device = torch.device('cuda')
    model = CRATOModel().to(device)
    
    # Random batch simulating canonical phase
    x = torch.randn(8, 140, device=device)
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, rho_t, I_t, P_t, P_hat_t, M_t = model(x)
        print(f"logits has nan: {torch.isnan(logits).any().item()}")
        
        # Test backward
        loss = logits.sum()
        loss.backward()
        print("Backward pass completed without crash.")
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
        
        loss = torch.sum(logits) + torch.sum(P_hat_t)
        
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN Gradient found in: {name}")

if __name__ == "__main__":
    test_nans()
