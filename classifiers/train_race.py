import os
import torch
import clip
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight

def load_data(base_path):
    race_folders = ["Asian", "Black", "Indian", "Others", "White"]
    image_paths, labels = [], []
    for idx, folder in enumerate(race_folders):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path): continue
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(folder_path, file))
                labels.append(idx)
    return image_paths, labels

class RaceDataset(Dataset):
    def __init__(self, paths, labels, transform, hard_label_aug_idx=[3,4]):
        self.paths, self.labels = paths, labels
        self.transform = transform
        self.hard_label_aug_idx = hard_label_aug_idx
        self.whiteaug = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomRotation(18),           # 强化旋转
            transforms.ColorJitter(0.22, 0.22, 0.22, 0.10), # 更重的光色变化
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711]),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]
        if label in self.hard_label_aug_idx:
            img = self.whiteaug(img)
        else:
            img = self.transform(img)
        return img, label

class RaceClassifier(nn.Module):
    def __init__(self, clip_visual, num_classes=5):
        super().__init__()
        self.visual = clip_visual
        for param in self.visual.parameters():
            param.requires_grad = False
        for block in self.visual.transformer.resblocks[-3:]:
            for p in block.parameters():
                p.requires_grad = True
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.visual(x)
        return self.head(features)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, input, target):
        ce_loss = nn.functional.cross_entropy(input, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def per_class_accuracy(true_labels, pred_labels, num_classes, phase, race_folders):
    total_per_class = [0] * num_classes
    correct_per_class = [0] * num_classes
    for t, p in zip(true_labels, pred_labels):
        total_per_class[t] += 1
        if t == p:
            correct_per_class[t] += 1
    print(f"{phase}阶段各类别准确数/总数:")
    for idx in range(num_classes):
        acc = correct_per_class[idx] / total_per_class[idx] if total_per_class[idx] > 0 else 0.0
        print(f"  {race_folders[idx]}: {correct_per_class[idx]}/{total_per_class[idx]} ({acc:.4f})")
    return correct_per_class, total_per_class

def main():
    base_path = r"/root/autodl-tmp/newfairface"
    batch_size = 64
    num_epochs = 30
    patience = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    race_folders = ["Asian", "Black", "Indian", "Others", "White"]
    num_classes = len(race_folders)

    image_paths, labels = load_data(base_path)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(0.09, 0.09, 0.09, 0.025),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # ---- 采样权重调整 ----
    np_labels = np.array(train_labels)
    weights = np.ones(len(np_labels))
    # Others和White采样提升至3.5
    for idx in [3,4]:       # 3: Others, 4: White
        weights[np_labels==idx] = 3.5
    sampler = WeightedRandomSampler(weights, num_samples=len(np_labels), replacement=True)

    train_dataset = RaceDataset(train_paths, train_labels, train_transform)
    val_dataset = RaceDataset(val_paths, val_labels, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model, preprocess = clip.load('ViT-B/16', device=device)
    classifier = RaceClassifier(model.visual).to(device)
    classifier = classifier.float()

    # FocalLoss, gamma=2.5
    criterion = FocalLoss(alpha=class_weights, gamma=2.5)
    optimizer = optim.AdamW([
        {"params": classifier.visual.transformer.resblocks[-3:].parameters(), "lr": 1e-5},
        {"params": classifier.head.parameters(), "lr": 5e-5}
    ], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_acc, early_stop_counter = 0, 0

    for epoch in range(num_epochs):
        classifier.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_true_labels, train_pred_labels = [], []
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = outputs.argmax(1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(pred.cpu().numpy())
            train_bar.set_postfix(loss=loss.item())

        classifier.eval()
        val_loss, correct, total = 0.0, 0, 0
        val_true_labels, val_pred_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = classifier(imgs)
                val_loss += criterion(preds, labels).item()
                pred = preds.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                val_true_labels.extend(labels.cpu().numpy())
                val_pred_labels.extend(pred.cpu().numpy())

        val_acc = correct / total
        scheduler.step(val_acc)
        print(f"Epoch {epoch + 1}: Train Acc {train_correct/train_total:.4f}, Train Loss {train_loss/len(train_loader):.4f}, Val Acc {val_acc:.4f}")
        per_class_accuracy(train_true_labels, train_pred_labels, num_classes, phase="训练", race_folders=race_folders)
        per_class_accuracy(val_true_labels, val_pred_labels, num_classes, phase="验证", race_folders=race_folders)

        if val_acc > best_acc:
            best_acc = val_acc
            early_stop_counter = 0
            torch.save(classifier.state_dict(), 'race_ViT_FT.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered!")
                break
    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()