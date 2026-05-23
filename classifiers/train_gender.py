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
from PIL import Image
import warnings
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# =============================================================
# ==                     CLASS DEFINITIONS                     ==
# =============================================================

class WaveletDetailTransform:
    """
    Extracts multi-scale structural features using discrete wavelet transform (DWT2)
    and stacks different sub-bands into channel dimensions.
    """
    def __init__(self, wavelet='db4'):
        self.wavelet = wavelet
        self.resizer = transforms.Resize((224, 224), antialias=True)

    def __call__(self, img_tensor):
        if not isinstance(img_tensor, torch.Tensor): 
            raise TypeError("Input must be a torch.Tensor")
        gray = transforms.functional.rgb_to_grayscale(img_tensor, num_output_channels=1) if img_tensor.shape[0] == 3 else img_tensor
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


class DINOv2AttentionWaveletClassifier(nn.Module):
    """
    Dual-stream classifier based on DINOv2 that joint-processes spatial and wavelet domain 
    information, fusing them via a learnable attention weight.
    """
    def __init__(self, num_classes, model_path="facebook/dinov2-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path)
        self.hidden_dim = self.backbone.config.hidden_size
        self.fusion_weight = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 1024), 
            nn.BatchNorm1d(1024), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(), 
            nn.Dropout(0.4), 
            nn.Linear(512, num_classes)
        )
        
    def forward(self, spatial_pixel_values, wavelet_pixel_values):
        batch_size = spatial_pixel_values.shape[0]
        combined_input = torch.cat([spatial_pixel_values, wavelet_pixel_values], dim=0)
        combined_out = self.backbone(pixel_values=combined_input)
        combined_cls = combined_out.last_hidden_state[:, 0]
        spatial_cls, wavelet_cls = combined_cls[:batch_size], combined_cls[batch_size:]
        w = torch.sigmoid(self.fusion_weight)
        fused = torch.cat([w * spatial_cls, (1 - w) * wavelet_cls], dim=-1)
        return self.classifier(fused)


class FocalLoss(nn.Module):
    """
    Focal Loss function to address the class imbalance issue.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None: 
            loss *= self.alpha[targets]
        return loss.mean() if self.reduction == "mean" else loss.sum()


class CustomDualStreamDataset(Dataset):
    """
    Real-time dataset wrapper that dynamically generates spatial and wavelet domain 
    tensors upon each item invocation.
    """
    def __init__(self, image_folder_dataset, base_transform, freq_transform, normalize_transform):
        self.image_folder_dataset = image_folder_dataset
        self.base_transform = base_transform
        self.freq_transform = freq_transform
        self.normalize = normalize_transform

    def __getitem__(self, index):
        try:
            img_pil, label = self.image_folder_dataset[index]
            spatial_tensor = self.base_transform(img_pil)
            freq_tensor = self.freq_transform(spatial_tensor)
            spatial_tensor_normalized = self.normalize(spatial_tensor)
            freq_tensor_normalized = self.normalize(freq_tensor)
            return (spatial_tensor_normalized, freq_tensor_normalized), label
        except Exception as e:
            print(f"Error processing image at index {index}: {e}")
            blank_spatial = torch.zeros((3, 224, 224))
            blank_freq = torch.zeros((3, 224, 224))
            return (blank_spatial, blank_freq), 0

    def __len__(self):
        return len(self.image_folder_dataset)

# =============================================================
# ==                    UTILITY FUNCTIONS                      ==
# =============================================================

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, filename="training_curves_gender_amp.png"):
    """Saves loss and accuracy performance curves for both training and validation phases."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Train Acc')
    plt.plot(epochs, val_accs, 'ro-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """Trains the model for one epoch using Automatic Mixed Precision (AMP)."""
    model.train()
    total_loss, correct = 0, 0
    for (spatial_x, freq_x), y in tqdm(loader, desc="Training with AMP"):
        spatial_x, freq_x, y = spatial_x.to(device), freq_x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with autocast():
            out = model(spatial_x, freq_x)
            loss = criterion(out, y)
        
        if torch.isnan(loss): 
            continue
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    """Evaluates the model performance utilizing Test-Time Augmentation (TTA) and autocast."""
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    for (spatial_x, freq_x), y in tqdm(loader, desc="Evaluating with TTA"):
        spatial_x, freq_x, y = spatial_x.to(device), freq_x.to(device), y.to(device)
        
        with autocast():
            out_original = model(spatial_x, freq_x)
            spatial_x_flipped = transforms.functional.hflip(spatial_x)
            out_flipped = model(spatial_x_flipped, freq_x)
            out_avg = (out_original + out_flipped) / 2.0
        
        loss = criterion(out_avg, y)
        total_loss += loss.item()
        pred = out_avg.argmax(1)
        correct += (pred == y).sum().item()
        all_preds.append(pred.cpu())
        all_labels.append(y.cpu())
        
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    print("\nClassification Report with TTA:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
    return total_loss / len(loader), correct / len(loader.dataset)

# =============================================================
# ==                      MAIN FUNCTION                        ==
# =============================================================
def main():
    # --- 0. Configurations ---
    DATA_DIR = r"/root/autodl-tmp/newgender"
    MODEL_PATH = "/root/autodl-tmp/dinov2-base"
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MAX_PATIENCE = 50, 64, 2e-5, 2e-2, 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Data Transforms ---
    train_spatial_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandAugment(), 
        transforms.ToTensor()
    ])
    val_spatial_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])
    secondary_transform = WaveletDetailTransform(wavelet='db4')
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # --- 2. Dataset Loading and Splitting ---
    print("Loading raw dataset via ImageFolder...")
    full_dataset_raw = datasets.ImageFolder(DATA_DIR)
    class_names, num_classes = list(full_dataset_raw.class_to_idx.keys()), len(full_dataset_raw.classes)
    print(f"Found {num_classes} classes: {class_names}")

    train_full_dataset = CustomDualStreamDataset(full_dataset_raw, train_spatial_transform, secondary_transform, normalize_transform)
    val_full_dataset = CustomDualStreamDataset(full_dataset_raw, val_spatial_transform, secondary_transform, normalize_transform)
    print("Real-time pipeline dataset initialization complete.")
    
    print("Performing stratified dataset splitting...")
    labels = np.array(full_dataset_raw.targets)
    train_idx, val_idx = [], []
    for class_label in range(num_classes):
        class_indices = np.where(labels == class_label)[0]
        np.random.shuffle(class_indices)
        val_size = int(len(class_indices) * 0.2)
        val_idx.extend(class_indices[:val_size])
        train_idx.extend(class_indices[val_size:])
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    print(f"Splitting complete. Train set size: {len(train_idx)}, Val set size: {len(val_idx)}")
    train_set, val_set = Subset(train_full_dataset, train_idx), Subset(val_full_dataset, val_idx)
    
    num_workers_to_use = 16 
    print(f"Utilizing {num_workers_to_use} data loader workers.")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers_to_use, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=num_workers_to_use, pin_memory=True)

    # --- 3. Model Init & 4. Criterion, Optimizer, Scheduler ---
    model = DINOv2AttentionWaveletClassifier(num_classes=num_classes, model_path=MODEL_PATH).to(device)
    print("Model backbone initialization complete.")
    
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = torch.tensor([len(train_labels) / (c + 1e-6) for c in class_counts], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True, min_lr=1e-7)
    
    scaler = GradScaler()
    
    # --- 5. Training Loop ---
    best_acc, patience = 0.0, 0
    print(f"Starting pipeline with AMP and {num_workers_to_use} workers for {NUM_EPOCHS} epochs...")
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, class_names)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1} Metrics Summary (Time elapsed: {epoch_duration:.2f}s):")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy (with TTA): {val_acc:.4f}")
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current Learning Rate: {current_lr:.8f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "DINO_gender_wav_w.pth")
            print(f"  --> Saved new best checkpoint with accuracy: {best_acc:.4f}")
            patience = 0
        else:
            patience += 1
            print(f"  Validation accuracy did not improve. Patience: {patience}/{MAX_PATIENCE}")
            if patience >= MAX_PATIENCE:
                print("Early stopping triggered!")
                break
        
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, filename="training_curves_gender_amp.png")

    print("\nTraining workflow fully completed!")
    print(f"Best Validation Accuracy achieved (with TTA): {best_acc:.4f}")


if __name__ == "__main__":
    main()
