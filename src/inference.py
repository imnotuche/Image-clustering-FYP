import torch
import joblib
import numpy as np
from dec_model import DEC

def run_inference(dataloader, feature_extractor, device, model_path, pca_model_path, limit=1000):
    # 1. Load the model expecting 2048 (matches your saved weights)
    dec_model = DEC(n_clusters=10, embedding_dim=2048).to(device)
    dec_model.load_state_dict(torch.load(model_path, map_location=device))
    dec_model.eval()
    
    feature_extractor.to(device)
    feature_extractor.eval()
    
    all_preds = []
    all_images = []
    total_count = 0

    print(f"Running inference on {limit} unseen images (Direct 2048-dim mode)...")
    with torch.no_grad():
        for images, _ in dataloader:
            if total_count >= limit:
                break
                
            images = images.to(device)
            
            # Step A: ResNet (Outputs 2048-dim)
            raw_feats = feature_extractor(images)
            
            # Step B: SKIP PCA 
            # We don't use 'pca.transform' because the DEC model wants the full 2048
            
            # Step C: DEC Prediction
            q = dec_model(raw_feats) # Use raw_feats directly
            preds = torch.argmax(q, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_images.append(images.cpu())
            
            total_count += images.size(0)

    final_preds = np.concatenate(all_preds)[:limit]
    final_images = torch.cat(all_images)[:limit]

    return final_preds, final_images