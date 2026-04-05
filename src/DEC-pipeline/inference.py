import torch
import torch.nn.functional as F
import numpy as np
from dec_model import DEC
import joblib

def run_inference(dataloader, feature_extractor, device, model, pca_model_path, limit=None):
    # 1. Load the model expecting 2048 (matches your saved weights)
    dec_model = DEC(n_clusters=10, embedding_dim=50).to(device)
    dec_model.load_state_dict(model)
    dec_model.eval()
    
    pca=joblib.load(pca_model_path)
    
    feature_extractor.to(device)
    feature_extractor.eval()
    
    all_preds = []
    all_images = []
    all_probs=[]
    total_count = 0

    print(f"Running inference on {limit} unseen images (Direct 2048-dim mode)...")
    with torch.no_grad():
        for images, _ in dataloader:
            if limit is not None and total_count >= limit:
                break
                
            images = images.to(device)
            
            # Step A: ResNet (Outputs 2048-dim)
            raw_feats = feature_extractor(images)
            raw_feats= F.normalize(raw_feats, p=2, dim=1)
            
            # Step B: PCA 
            reduced_feats=pca.transform(raw_feats)
            reduced_feats=torch.from_numpy(reduced_feats).float().to(device)
            
            # Step C: DEC Prediction
            q = dec_model(reduced_feats)
            preds = torch.argmax(q, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_images.append(images.cpu())
            all_probs.append(q.cpu().numpy())
            
            total_count += images.size(0)

    if limit is not None:
        final_preds = np.concatenate(all_preds) [:limit]
        final_images = torch.cat(all_images)[:limit]
        final_probs = np.concatenate(all_probs)[:limit]
    else:
        final_preds = np.concatenate(all_preds)
        final_images = torch.cat(all_images)
        final_probs = np.concatenate(all_probs)

    return final_preds, final_images, final_probs