"""
Plant Disease Prediction Model
================================
Architecture : EfficientNetB3 (Transfer Learning) with custom classification head
Dataset      : PlantVillage-style (38 disease classes across 14 plant species)
Approach     : Fine-tuned CNN + Data Augmentation + Grad-CAM Visualization

Best-suited algorithm justification:
- EfficientNetB3 achieves ~98% accuracy on PlantVillage with far fewer params than ResNet/VGG
- Transfer learning from ImageNet captures low-level texture/edge features reusable for leaves
- Progressive unfreezing avoids catastrophic forgetting during fine-tuning
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    # Paths
    DATA_DIR        = "New Plant Diseases Dataset(Augmented)" # Root dataset folder
    MODEL_SAVE_PATH = "checkpoints/best_model.pth"
    LOG_DIR         = "logs"
    RESULTS_DIR     = "results"

    # Model
    MODEL_NAME      = "efficientnet_b3"            # backbone
    NUM_CLASSES     = 38                            # PlantVillage has 38 classes
    PRETRAINED      = True

    # Training
    IMG_SIZE        = 224
    BATCH_SIZE      = 32
    EPOCHS          = 30
    LR              = 1e-3                          # initial LR for classifier head
    LR_BACKBONE     = 1e-5                          # fine-tune LR for backbone
    WEIGHT_DECAY    = 1e-4
    VAL_SPLIT       = 0.15
    TEST_SPLIT      = 0.10
    SEED            = 42
    NUM_WORKERS     = 4
    UNFREEZE_EPOCH  = 5                             # epoch to start fine-tuning backbone
    EARLY_STOP_PAT  = 7                             # early stopping patience
    DROPOUT         = 0.4

    # Device
    DEVICE          = "cuda" if torch.cuda.is_available() else (
                      "mps"  if torch.backends.mps.is_available() else "cpu")


cfg = Config()
os.makedirs(cfg.LOG_DIR,     exist_ok=True)
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(cfg.MODEL_SAVE_PATH), exist_ok=True)
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
print(f"[Config] Device: {cfg.DEVICE}  |  Model: {cfg.MODEL_NAME}  |  Classes: {cfg.NUM_CLASSES}")


# ─────────────────────────────────────────────
#  DATA AUGMENTATION & TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms(phase: str) -> transforms.Compose:
    """
    Training  : aggressive augmentation to combat overfitting on leaf images.
    Validation: only resize + normalize (no augmentation = fair evaluation).
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats
    std  = [0.229, 0.224, 0.225]

    if phase == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2),          # Cutout regularisation
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─────────────────────────────────────────────
#  DATASET UTILITY
# ─────────────────────────────────────────────
def load_datasets(data_dir: str):
    """
    Loads separate train and valid folders.
    Splits 'valid' into validation and test sets.
    """
    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "valid")

    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        raise FileNotFoundError(f"Dataset structure not found in {data_dir}. "
                                "Expected 'train' and 'valid' folders.")

    train_ds = ImageFolder(root=train_path, transform=get_transforms("train"))
    class_names = train_ds.classes

    # Load validation folder
    valid_full_ds = ImageFolder(root=valid_path, transform=get_transforms("val"))

    # Split valid folder into validation and test sets
    n_valid_total = len(valid_full_ds)
    n_test = int(n_valid_total * 0.5)  # Split valid in half for test
    n_val  = n_valid_total - n_test

    val_ds, test_ds = random_split(
        valid_full_ds, [n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.SEED)
    )

    print(f"[Dataset] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    print(f"[Dataset] Classes ({len(class_names)}): {class_names[:5]} ...")
    return train_ds, val_ds, test_ds, class_names


def get_loaders(train_ds, val_ds, test_ds):
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class PlantDiseaseClassifier(nn.Module):
    """
    EfficientNetB3 backbone with a custom multi-layer classification head.

    Architecture choice rationale:
    ┌─────────────────────────────────────────────────────────┐
    │  EfficientNetB3 vs alternatives                         │
    │  ─────────────────────────────────────────────────────  │
    │  VGG16        : 138M params, ~90% acc  → too heavy      │
    │  ResNet50     : 25M params,  ~95% acc  → good baseline  │
    │  DenseNet121  : 8M params,   ~95% acc  → good           │
    │  EfficientNetB3: 12M params, ~98% acc  → BEST tradeoff  │
    │                                                         │
    │  EfficientNet uses compound scaling (depth+width+res)   │
    │  which optimally scales for image classification tasks. │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # Load backbone
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        # Extract feature extractor (everything except classifier)
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool
        in_features     = backbone.classifier[1].in_features   # 1536 for B3

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),                          # Swish activation (used in EfficientNet)
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(cfg.DROPOUT / 2),
            nn.Linear(256, num_classes),
        )

        # Freeze backbone initially
        self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Progressively unfreeze deeper layers for fine-tuning."""
        for p in self.features.parameters():
            p.requires_grad = True
        print("[Model] Backbone unfrozen for fine-tuning.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x: torch.Tensor):
        """Returns penultimate feature maps for Grad-CAM."""
        return self.features(x)

    def count_parameters(self):
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Total params: {total:,}  |  Trainable: {trainable:,}")


# ─────────────────────────────────────────────
#  TRAINING ENGINE
# ─────────────────────────────────────────────
class Trainer:
    def __init__(self, model, train_loader, val_loader, class_names):
        self.model        = model.to(cfg.DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.class_names  = class_names
        self.history      = {"train_loss": [], "val_loss": [],
                             "train_acc":  [], "val_acc":  []}
        self.best_val_acc = 0.0
        self.no_improve   = 0

        # Class-weighted loss to handle imbalanced data
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Two param groups: classifier (high LR) + backbone (low LR after unfreeze)
        self.optimizer = optim.AdamW([
            {"params": model.classifier.parameters(), "lr": cfg.LR},
            {"params": model.features.parameters(),   "lr": cfg.LR_BACKBONE},
        ], weight_decay=cfg.WEIGHT_DECAY)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
        self.scaler    = torch.cuda.amp.GradScaler(enabled=(cfg.DEVICE == "cuda"))

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(cfg.DEVICE == "cuda")):
                outputs = self.model(imgs)
                loss    = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            outputs = self.model(imgs)
            loss    = self.criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        return total_loss / total, correct / total

    def fit(self):
        print(f"\n{'-'*60}")
        print(f"  Training Plant Disease Classifier  |  {cfg.EPOCHS} epochs")
        print(f"{'-'*60}")
        self.model.count_parameters()

        for epoch in range(1, cfg.EPOCHS + 1):
            # Progressive unfreeze at UNFREEZE_EPOCH
            if epoch == cfg.UNFREEZE_EPOCH:
                self.model.unfreeze_backbone()

            t_loss, t_acc = self.train_epoch()
            v_loss, v_acc = self.val_epoch()
            self.scheduler.step()

            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            self.history["train_acc"].append(t_acc)
            self.history["val_acc"].append(v_acc)

            # Checkpoint
            if v_acc > self.best_val_acc:
                self.best_val_acc = v_acc
                self.no_improve   = 0
                torch.save({"epoch": epoch,
                            "model_state": self.model.state_dict(),
                            "class_names": self.class_names,
                            "val_acc": v_acc},
                           cfg.MODEL_SAVE_PATH)
                flag = "✓ saved"
            else:
                self.no_improve += 1
                flag = f"(no improve {self.no_improve}/{cfg.EARLY_STOP_PAT})"

            print(f"  Epoch {epoch:3d}/{cfg.EPOCHS}  "
                  f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f}  │  "
                  f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}  {flag}")

            # Early stopping
            if self.no_improve >= cfg.EARLY_STOP_PAT:
                print(f"\n[Early Stop] No improvement for {cfg.EARLY_STOP_PAT} epochs. Stopping.")
                break

        print(f"\n[Training Complete] Best Val Accuracy: {self.best_val_acc:.4f}")
        return self.history


# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, test_loader, class_names, save_dir=cfg.RESULTS_DIR):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in test_loader:
        imgs   = imgs.to(cfg.DEVICE)
        preds  = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    print("\n" + "="*60)
    print("  CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt="d", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix – Plant Disease Prediction", fontsize=16, pad=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0,  fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150)
    print(f"[Eval] Confusion matrix saved -> {save_dir}/confusion_matrix.png")


def plot_training_history(history, save_dir=cfg.RESULTS_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History – Plant Disease Classifier", fontsize=14)

    axes[0].plot(history["train_loss"], label="Train", color="#2ecc71", lw=2)
    axes[0].plot(history["val_loss"],   label="Val",   color="#e74c3c", lw=2)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", color="#2ecc71", lw=2)
    axes[1].plot(history["val_acc"],   label="Val",   color="#e74c3c", lw=2)
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_history.png", dpi=150)
    print(f"[Plot] Training history saved -> {save_dir}/training_history.png")


# ─────────────────────────────────────────────
#  GRAD-CAM VISUALIZATION
# ─────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Highlights the leaf regions the model focuses on for its prediction.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.gradients    = None
        self.activations  = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor: torch.Tensor, class_idx: int = None):
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0).to(cfg.DEVICE)
        output     = self.model(img_tensor)
        if class_idx is None:
            class_idx = output.argmax(1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def visualize_gradcam(model, img_path: str, class_names: list, save_path: str):
    """Overlay Grad-CAM heatmap on the original leaf image."""
    # Target the last conv block of EfficientNetB3
    target_layer = model.features[-1]
    gcam = GradCAM(model, target_layer)

    transform = get_transforms("val")
    img_pil    = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil)

    cam, pred_idx = gcam.generate(img_tensor)
    cam_resized   = np.array(Image.fromarray(cam).resize(
                    img_pil.size, Image.BILINEAR))

    heatmap = cm.jet(cam_resized)[..., :3]
    overlay = 0.5 * np.array(img_pil) / 255.0 + 0.5 * heatmap

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(img_pil);              axes[0].set_title("Original Leaf")
    axes[1].imshow(cam_resized, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap")
    axes[2].imshow(np.clip(overlay, 0, 1)); axes[2].set_title(
        f"Prediction: {class_names[pred_idx]}")
    for ax in axes: ax.axis("off")
    plt.suptitle("Grad-CAM Explainability", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Grad-CAM] Visualization saved -> {save_path}")
    return class_names[pred_idx]


# ─────────────────────────────────────────────
#  INFERENCE PIPELINE
# ─────────────────────────────────────────────
class PlantDiseasePredictor:
    """
    Production-ready inference wrapper.
    Usage:
        predictor = PlantDiseasePredictor("checkpoints/best_model.pth")
        result    = predictor.predict("leaf.jpg")
        print(result)
    """

    def __init__(self, checkpoint_path: str):
        ckpt         = torch.load(checkpoint_path, map_location=cfg.DEVICE)
        self.classes = ckpt["class_names"]
        self.model   = PlantDiseaseClassifier(num_classes=len(self.classes),
                                              pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(cfg.DEVICE).eval()
        self.transform = get_transforms("val")
        print(f"[Predictor] Loaded model. Classes: {len(self.classes)}")

    @torch.no_grad()
    def predict(self, img_path: str, top_k: int = 5) -> dict:
        img    = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(cfg.DEVICE)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        top_probs, top_idx = probs.topk(top_k)

        results = {
            "predicted_class":      self.classes[top_idx[0].item()],
            "confidence":           round(top_probs[0].item() * 100, 2),
            "top_k_predictions": [
                {"class": self.classes[i.item()],
                 "probability": round(p.item() * 100, 2)}
                for i, p in zip(top_idx, top_probs)
            ]
        }
        return results

    def predict_batch(self, img_paths: list) -> list:
        return [self.predict(p) for p in img_paths]


# ─────────────────────────────────────────────
#  MAIN  ─  full training pipeline
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("     PLANT DISEASE PREDICTION SYSTEM")
    print("      EfficientNetB3 + Transfer Learning")
    print("="*60 + "\n")

    # 1. Data
    train_ds, val_ds, test_ds, class_names = load_datasets(cfg.DATA_DIR)
    train_loader, val_loader, test_loader  = get_loaders(train_ds, val_ds, test_ds)

    # Save class mapping
    with open(f"{cfg.LOG_DIR}/class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # 2. Model
    model = PlantDiseaseClassifier(num_classes=len(class_names),
                                   pretrained=cfg.PRETRAINED)

    # 3. Train
    trainer = Trainer(model, train_loader, val_loader, class_names)
    history = trainer.fit()
    plot_training_history(history)

    # 4. Evaluate on test set
    ckpt  = torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    evaluate(model, test_loader, class_names)

    print(f"\n[Done] All artifacts saved in '{cfg.RESULTS_DIR}/' and '{cfg.LOG_DIR}/'")
    print("[Usage] See PlantDiseasePredictor class for inference on new images.\n")


if __name__ == "__main__":
    main()
