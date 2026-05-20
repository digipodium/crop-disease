# 🌿 Plant Disease Prediction System

## Architecture: EfficientNetB3 + Transfer Learning

### Why EfficientNetB3?

| Model | Params | Val Accuracy | Notes |
|---|---|---|---|
| VGG16 | 138M | ~90% | Too heavy, overfits |
| ResNet50 | 25M | ~95% | Good baseline |
| DenseNet121 | 8M | ~95% | Memory intensive |
| **EfficientNetB3** | **12M** | **~98%** | **Best balance** |

EfficientNet uses **compound scaling** — balancing depth, width, and resolution together — which is optimal for fine-grained leaf texture classification.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the dataset
```bash
# Option A: Check dataset stats for the augmented dataset
python dataset_setup.py --stats
```

### 3. Train
You can train the model in two ways:

#### Option A: Interactive Jupyter Notebook (Recommended)
Launch the interactive Jupyter notebook [plant_disease_training.ipynb](file:///c:/Users/tripl/Downloads/new_plant/plant_disease_training.ipynb) to train the model, view custom data visualizations, generate detailed performance evaluation metrics, and explore Grad-CAM explainability maps step-by-step:
```bash
jupyter notebook plant_disease_training.ipynb
```

#### Option B: CLI Script
Run the script to train the model through the command-line interface:
```bash
python plant_disease_model.py
```

### 4. Predict on new images
```python
from plant_disease_model import PlantDiseasePredictor

predictor = PlantDiseasePredictor("checkpoints/best_model.pth")
result    = predictor.predict("my_leaf.jpg")
print(result)
# {
#   "predicted_class": "Tomato___Late_blight",
#   "confidence": 97.42,
#   "top_k_predictions": [...]
# }
```

### 5. Grad-CAM explainability
```python
from plant_disease_model import visualize_gradcam, PlantDiseaseClassifier
import torch, json

class_names = json.load(open("logs/class_names.json"))
model = PlantDiseaseClassifier(len(class_names), pretrained=False)
ckpt  = torch.load("checkpoints/best_model.pth")
model.load_state_dict(ckpt["model_state"])

visualize_gradcam(model, "my_leaf.jpg", class_names, "results/gradcam.png")
```

### 6. Flask web app integration
Run the Flask web app to add user registration, login, image upload, and prediction result pages.

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

- Register a new account
- Login
- Upload a leaf image (jpg/jpeg/png)
- View prediction and confidence results

---

## Dataset: New Plant Diseases Dataset (Augmented)
- **87,867** images across **38 disease classes** (70,295 training, 17,572 validation)
- **14 plant species**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- Covers both **diseased** and **healthy** leaves

---

## Training Pipeline

```
Raw Leaf Images
      │
      ▼
Data Augmentation (RandomCrop, Flip, Rotation, ColorJitter, RandomErasing)
      │
      ▼
EfficientNetB3 Backbone (ImageNet pretrained, frozen)
      │
      ▼ [Epoch 5: unfreeze backbone for fine-tuning]
      │
Custom Head: Linear(1536→512) → BN → SiLU → Dropout(0.4)
           → Linear(512→256)  → BN → SiLU → Dropout(0.2)
           → Linear(256→38)
      │
      ▼
CrossEntropyLoss (label smoothing=0.1)
AdamW + CosineAnnealingLR
Early Stopping (patience=7)
```

---

## Output Files

```
checkpoints/
  best_model.pth        ← best checkpoint (by val accuracy)
logs/
  class_names.json      ← class index mapping
results/
  training_history.png  ← loss & accuracy curves
  confusion_matrix.png  ← per-class confusion matrix
  gradcam.png           ← Grad-CAM visualization
```

---

## Expected Performance (Augmented Dataset)

| Metric | Value |
|---|---|
| Test Accuracy | ~97–98% |
| Training Time (GPU) | ~45 min |
| Training Time (CPU) | ~8 hours |
| Model Size | ~48 MB |
| Inference Speed | ~5ms/image (GPU) |
