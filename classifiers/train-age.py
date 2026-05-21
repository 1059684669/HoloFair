import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pywt


# =============================================================
# ==                  CLASS DEFINITIONS                      ==
# =============================================================

class WaveletDetailTransform:

    def __init__(self, wavelet='db4'):
        self.wavelet = wavelet
        self.resizer = transforms.Resize((224, 224), antialias=True)

    def __call__(self, img_tensor):
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        gray = transforms.functional.rgb_to_grayscale(img_tensor, num_output_channels=1) if img_tensor.shape[
                                                                                                0] == 3 else img_tensor
        gray_np = gray.squeeze(0).cpu().numpy()


        coeffs = pywt.dwt2(gray_np, self.wavelet)
        cA, (cH, cV, cD) = coeffs

        (h, w) = cA.shape
        cH, cV, cD = cH[:h, :w], cV[:h, :w], cD[:h, :w]


        cA = (cA - cA.min()) / (cA.max() - cA.min() + 1e-6)
        cH = (cH - cH.min()) / (cH.max() - cH.min() + 1e-6)
        cV = (cV - cV.min()) / (cV.max() - cV.min() + 1e-6)


        cA_tensor = torch.from_numpy(cA).float().unsqueeze(0)
        cH_tensor = torch.from_numpy(cH).float().unsqueeze(0)
        cV_tensor = torch.from_numpy(cV).float().unsqueeze(0)


        combined_tensor = torch.cat([cA_tensor, cH_tensor, cV_tensor], dim=0)


        resized_tensor = self.resizer(combined_tensor.unsqueeze(0)).squeeze(0)
        return resized_tensor


# Model, Loss, Dataset, and other utility classes/functions remain the same
# The DINOv2AttentionWaveletClassifier is general enough to handle any (3, 224, 224) input
class DINOv2AttentionWaveletClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        print("Initializing DINOv2AttentionWaveletClassifier...")
        self.backbone = AutoModel.from_pretrained("/root/autodl-tmp/dinov2-base")
        self.hidden_dim = self.backbone.config.hidden_size
        self.fusion_weight = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, spatial_pixel_values, wavelet_pixel_values):
        batch_size = spatial_pixel_values.shape[0]
        combined_input = torch.cat([spatial_pixel_values, wavelet_pixel_values], dim=0)
        combined_out = self.backbone(pixel_values=combined_input)
        combined_cls = combined_out.last_hidden_state[:, 0]
        spatial_cls = combined_cls[:batch_size]
        wavelet_cls = combined_cls[batch_size:]
        w = torch.sigmoid(self.fusion_weight)
        weighted_spatial_cls = w * spatial_cls
        weighted_wavelet_cls = (1 - w) * wavelet_cls
        fused = torch.cat([weighted_spatial_cls, weighted_wavelet_cls], dim=-1)
        return self.classifier(fused)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, reduction="mean"):
        super().__init__();
        self.alpha = alpha;
        self.gamma = gamma;
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none');
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None: loss *= self.alpha[targets]
        return loss.mean() if self.reduction == "mean" else loss.sum()


class CustomAgeDataset(Dataset):
    def __init__(self, image_folder_dataset, base_transform, freq_transform, normalize_transform):
        self.image_folder_dataset = image_folder_dataset;
        self.base_transform = base_transform
        self.freq_transform = freq_transform;
        self.normalize = normalize_transform

    def __getitem__(self, index):
        img_pil, label = self.image_folder_dataset[index]
        spatial_tensor = self.base_transform(img_pil)
        freq_tensor = self.freq_transform(spatial_tensor)
        spatial_tensor = self.normalize(spatial_tensor);
        freq_tensor = self.normalize(freq_tensor)
        return (spatial_tensor, freq_tensor), label

    def __len__(self): return len(self.image_folder_dataset)


# =============================================================
# ==                  UTILITY FUNCTIONS                      ==
# =============================================================

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, filename="training_curves_ultimate.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1);
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss');
    plt.plot(epochs, val_losses, 'ro-', label='Val Loss');
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.title('Loss Curve');
    plt.legend();
    plt.grid(True)
    plt.subplot(1, 2, 2);
    plt.plot(epochs, train_accs, 'bo-', label='Train Acc');
    plt.plot(epochs, val_accs, 'ro-', label='Val Acc');
    plt.xlabel('Epoch');
    plt.ylabel('Accuracy');
    plt.title('Accuracy Curve');
    plt.legend();
    plt.grid(True)
    plt.tight_layout();
    plt.savefig(filename);
    plt.close()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train();
    total_loss, correct = 0, 0
    for (spatial_x, freq_x), y in tqdm(loader, desc="Training"):
        spatial_x, freq_x, y = spatial_x.to(device), freq_x.to(device), y.to(device)
        out = model(spatial_x, freq_x);
        loss = criterion(out, y)
        if torch.isnan(loss): print("Warning: NaN loss detected!"); continue
        optimizer.zero_grad();
        loss.backward();
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0);
        optimizer.step()
        total_loss += loss.item();
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval();
    total_loss, correct = 0, 0;
    all_preds, all_labels = [], []
    for (spatial_x, freq_x), y in tqdm(loader, desc="Evaluating with TTA"):
        spatial_x, freq_x, y = spatial_x.to(device), freq_x.to(device), y.to(device)
        out_original = model(spatial_x, freq_x)
        spatial_x_flipped = transforms.functional.hflip(spatial_x)
        out_flipped = model(spatial_x_flipped, freq_x)
        out_avg = (out_original + out_flipped) / 2.0
        loss = criterion(out_avg, y)
        total_loss += loss.item();
        pred = out_avg.argmax(1);
        correct += (pred == y).sum().item()
        all_preds.append(pred.cpu());
        all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds);
    all_labels = torch.cat(all_labels)
    print("\nClassification Report (with TTA):");
    print(classification_report(all_labels, all_preds, target_names=['elderly', 'middle', 'young'], digits=4,
                                zero_division=0))
    return total_loss / len(loader), correct / len(loader.dataset)


# =============================================================
# ==                     MAIN FUNCTION                       ==
# =============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Data Transforms ---
    print("Initializing transforms with RandAugment and the new WaveletDetailTransform.")
    train_spatial_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandAugment(), transforms.ToTensor()
    ])
    val_spatial_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor()
    ])


    secondary_transform = WaveletDetailTransform(wavelet='db4')

    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # --- 2. Data Loading and Splitting ---
    print("Loading dataset...");
    full_dataset_raw = datasets.ImageFolder("/root/autodl-tmp/oldage")
    # Pass the new transform to the dataset
    train_full_dataset = CustomAgeDataset(full_dataset_raw, train_spatial_transform, secondary_transform,
                                          normalize_transform)
    val_full_dataset = CustomAgeDataset(full_dataset_raw, val_spatial_transform, secondary_transform,
                                        normalize_transform)
    print("Dataset loading and wrapping complete.")
    print("Stratifying and splitting dataset...");
    labels = np.array(full_dataset_raw.targets);
    train_idx, val_idx = [], [];
    class_names = list(full_dataset_raw.class_to_idx.keys())
    for class_label in range(len(class_names)):
        class_indices = np.where(labels == class_label)[0];
        np.random.shuffle(class_indices);
        val_size = int(len(class_indices) * 0.2)
        val_idx.extend(class_indices[:val_size]);
        train_idx.extend(class_indices[val_size:])
    np.random.shuffle(train_idx);
    np.random.shuffle(val_idx)
    print(f"Dataset split complete. Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    train_set = Subset(train_full_dataset, train_idx);
    val_set = Subset(val_full_dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # --- 3. Model Initialization (with Attention) ---
    model = DINOv2AttentionWaveletClassifier(num_classes=len(class_names)).to(device)
    print("Model initialized without freezing any layers.")

    # --- 4. Loss, Optimizer, and Scheduler ---
    train_labels = labels[train_idx];
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor([len(train_labels) / (c + 1e-6) for c in class_counts], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=1.0)

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=2e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2, verbose=True, min_lr=1e-7)

    # --- 5. Training Loop ---
    num_epochs = 50;
    best_acc, patience = 0.0, 0;
    max_patience = 10
    print(f"Starting Ultimate Training Run (Wavelet Detail Stacking + Attention + TTA) for {num_epochs} epochs...")
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss);
        val_losses.append(val_loss);
        train_accs.append(train_acc);
        val_accs.append(val_acc)
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch + 1} Summary (Duration: {epoch_duration:.2f}s):")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc (TTA): {val_acc:.4f}")

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current Learning Rate: {current_lr:.8f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "age-dino+wavelet+attention.pth")
            print(f"  --> NEW BEST MODEL SAVED! Accuracy: {best_acc:.4f}")
            patience = 0
        else:
            patience += 1
            print(f"  Validation accuracy did not improve. Patience: {patience}/{max_patience}")
            if patience >= max_patience:
                print(f"Early stopping triggered after {max_patience} epochs with no improvement.");
                break

        plot_training_curves(train_losses, val_losses, train_accs, val_accs, "training_curves_wavelet_detail_run.png")

    print("\nTraining complete!")
    print(f"FINAL BEST VALIDATION ACCURACY: {best_acc:.4f}")


if __name__ == "__main__":
    main()