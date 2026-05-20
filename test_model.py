"""
Plant Disease Model - Testing & Evaluation Script
=================================================
Loads the pre-existing checkpoint (best_model.pth) and evaluates it on the test set,
generating the Classification Report and Confusion Matrix, as well as a Grad-CAM
explainability visualization.
"""

import os
import torch
import random
import json
from pathlib import Path
from plant_disease_model import (
    cfg,
    load_datasets,
    get_loaders,
    PlantDiseaseClassifier,
    evaluate,
    PlantDiseasePredictor,
    visualize_gradcam
)

def main():
    print("\n" + "="*60)
    print("      PLANT DISEASE MODEL TESTING SYSTEM")
    print("="*60 + "\n")

    # 1. Load the dataset
    print("[1/4] Loading dataset split...")
    train_ds, val_ds, test_ds, class_names = load_datasets(cfg.DATA_DIR)
    _, _, test_loader = get_loaders(train_ds, val_ds, test_ds)

    # 2. Initialize model and load checkpoint
    print(f"\n[2/4] Loading model checkpoint from '{cfg.MODEL_SAVE_PATH}'...")
    if not os.path.exists(cfg.MODEL_SAVE_PATH):
        print(f"Error: Model checkpoint not found at '{cfg.MODEL_SAVE_PATH}'.")
        print("Please make sure you have trained the model or placed the 'best_model.pth' file in the 'checkpoints' directory.")
        return

    ckpt = torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE)
    model = PlantDiseaseClassifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(cfg.DEVICE)
    print("[Model] Checkpoint loaded successfully!")

    # 3. Evaluate on the full test set
    print("\n[3/4] Running evaluation on the test set...")
    evaluate(model, test_loader, class_names)
    print(f"[Eval] Confusion matrix and evaluation reports saved to '{cfg.RESULTS_DIR}/'")

    # 4. Grad-CAM and Predictor Demo on a random validation image
    print("\n[4/4] Running Grad-CAM and Single Inference Demo...")
    valid_folder_path = os.path.join(cfg.DATA_DIR, "valid")
    all_images = (list(Path(valid_folder_path).rglob("*.jpg")) + 
                  list(Path(valid_folder_path).rglob("*.JPG")) + 
                  list(Path(valid_folder_path).rglob("*.png")))
    
    if all_images:
        random_img_path = str(random.choice(all_images))
        print(f"\nSelected random validation image: {random_img_path}")
        
        # Grad-CAM Attention map
        gradcam_save_path = f"{cfg.RESULTS_DIR}/gradcam_test.png"
        visualize_gradcam(model, random_img_path, class_names, save_path=gradcam_save_path)
        print(f"[Grad-CAM] Attention map visualization saved to '{gradcam_save_path}'")
        
        # Single Image Predictor inference
        predictor = PlantDiseasePredictor(cfg.MODEL_SAVE_PATH)
        result = predictor.predict(random_img_path)
        print("\nSingle Leaf Inference Result:")
        print(json.dumps(result, indent=2))
    else:
        print("No validation images found to run inference/Grad-CAM demonstration.")

if __name__ == "__main__":
    main()
