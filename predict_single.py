"""
Plant Disease Prediction - Single Image Inference
=================================================
Loads the trained model checkpoint (checkpoints/best_model.pth) and performs
inference on a single user-specified leaf image.

Usage:
    python predict_single.py --image <path_to_leaf_image>
"""

import os
import argparse
import json
from plant_disease_model import cfg, PlantDiseasePredictor

def main():
    parser = argparse.ArgumentParser(description="Predict Plant Disease from a single leaf image")
    parser.add_argument(
        "--image", 
        type=str, 
        required=True, 
        help="Path to the leaf image file (e.g., leaf.jpg)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=cfg.MODEL_SAVE_PATH, 
        help="Path to the model checkpoint file (default: checkpoints/best_model.pth)"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=5, 
        help="Number of top predictions to display (default: 5)"
    )
    
    args = parser.parse_args()

    # Verify image path
    if not os.path.exists(args.image):
        print(f"Error: Leaf image not found at '{args.image}'. Please check the path.")
        return

    # Verify checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint not found at '{args.checkpoint}'.")
        print("Please make sure you have trained the model and placed 'best_model.pth' in the 'checkpoints' directory.")
        return

    print(f"\n[Inference] Loading model from checkpoint: '{args.checkpoint}'...")
    try:
        predictor = PlantDiseasePredictor(args.checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"[Inference] Running prediction on image: '{args.image}'...")
    try:
        results = predictor.predict(args.image, top_k=args.top_k)
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # Display results
    print("\n" + "="*50)
    print("           INFERENCE RESULTS")
    print("="*50)
    
    # Beautifully format class names for readability
    pred_class = results["predicted_class"].replace("___", " - ").replace("_", " ")
    print(f"[*] Predicted Category: {pred_class}")
    print(f"[*] Confidence Score:  {results['confidence']}%")
    print("-" * 50)
    
    print(f"Top {args.top_k} Probabilities:")
    for idx, pred in enumerate(results["top_k_predictions"], 1):
        clean_name = pred["class"].replace("___", " - ").replace("_", " ")
        print(f"  {idx}. {clean_name:<45} {pred['probability']}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
