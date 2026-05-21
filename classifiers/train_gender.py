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

# =============================================================
# ==                     类定义 (CLASS DEFINITIONS)            ==
# =============================================================

class WaveletDetailTransform:
    """
    使用小波变换提取多尺度结构特征，并将不同分量堆叠到不同通道。
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
    一个使用DINOv2的双流分类器，它能同时处理空间域和小波域的信息，
    并通过一个可学习的注意力机制将它们融合。
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
    Focal Loss损失函数，用于解决类别不平衡问题。
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
    实时处理版本的数据集类，在每次调用时动态生成空间域和小波域张量。
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
            print(f"处理索引 {index} 的图像时出错: {e}")
            blank_spatial = torch.zeros((3, 224, 224))
            blank_freq = torch.zeros((3, 224, 224))
            return (blank_spatial, blank_freq), 0

    def __len__(self):
        return len(self.image_folder_dataset)

# =============================================================
# ==                   工具函数 (UTILITY FUNCTIONS)            ==
# =============================================================

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, filename="training_curves_gender_amp.png"):
    """保存训练和验证的损失及准确率曲线图。"""
    epochs=range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='训练损失')
    plt.plot(epochs, val_losses, 'ro-', label='验证损失')
    plt.xlabel('周期 (Epoch)'); plt.ylabel('损失 (Loss)')
    plt.title('损失曲线 (Loss Curve)'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='训练准确率')
    plt.plot(epochs, val_accs, 'ro-', label='验证准确率')
    plt.xlabel('周期 (Epoch)'); plt.ylabel('准确率 (Accuracy)')
    plt.title('准确率曲线 (Accuracy Curve)'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(filename); plt.close()

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """使用混合精度训练模型一个周期。"""
    model.train()
    total_loss, correct = 0, 0
    for (spatial_x, freq_x), y in tqdm(loader, desc="正在训练 (Training with AMP)"):
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
    """使用TTA和autocast评估模型。"""
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    for (spatial_x, freq_x), y in tqdm(loader, desc="正在评估 (Evaluating with TTA)"):
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
    print("\n分类报告 (Classification Report with TTA):")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
    return total_loss / len(loader), correct / len(loader.dataset)

# =============================================================
# ==                      主函数 (MAIN FUNCTION)               ==
# =============================================================
def main():
    # --- 0. 配置信息 ---
    # *** 关键改动：更新为您的性别分类数据集路径 ***
    DATA_DIR = r"/root/autodl-tmp/newgender"
    MODEL_PATH = "/root/autodl-tmp/dinov2-base"
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MAX_PATIENCE = 50, 64, 2e-5, 2e-2, 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # --- 1. 数据转换 ---
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

    # --- 2. 数据加载与分割 ---
    print("正在加载数据集...")
    full_dataset_raw = datasets.ImageFolder(DATA_DIR)
    class_names, num_classes = list(full_dataset_raw.class_to_idx.keys()), len(full_dataset_raw.classes)
    print(f"找到 {num_classes} 个类别: {class_names}")

    train_full_dataset = CustomDualStreamDataset(full_dataset_raw, train_spatial_transform, secondary_transform, normalize_transform)
    val_full_dataset = CustomDualStreamDataset(full_dataset_raw, val_spatial_transform, secondary_transform, normalize_transform)
    print("实时处理数据集加载和封装完成。")
    
    print("正在对数据集进行分层分割...")
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
    print(f"数据集分割完成。训练集大小: {len(train_idx)}, 验证集大小: {len(val_idx)}")
    train_set, val_set = Subset(train_full_dataset, train_idx), Subset(val_full_dataset, val_idx)
    
    # 增加 num_workers 的数量以加速数据加载
    num_workers_to_use = 16 
    print(f"使用 {num_workers_to_use} 个 workers 进行数据加载。")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers_to_use, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=num_workers_to_use, pin_memory=True)

    # --- 3. 模型初始化 & 4. 损失、优化器、调度器 ---
    model = DINOv2AttentionWaveletClassifier(num_classes=num_classes, model_path=MODEL_PATH).to(device)
    # 可选: 使用torch.compile()进一步加速 (需要PyTorch 2.0+)
    # if torch.__version__.startswith('2'):
    #     model = torch.compile(model)
    print("模型初始化完成。")
    
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = torch.tensor([len(train_labels) / (c + 1e-6) for c in class_counts], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True, min_lr=1e-7)
    
    # 初始化混合精度训练的 GradScaler
    scaler = GradScaler()
    
    # --- 5. 训练循环 ---
    best_acc, patience = 0.0, 0
    print(f"开始使用AMP和高num_workers进行训练，共 {NUM_EPOCHS} 个周期...")
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        print(f"\n--- 周期 {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, class_names)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        epoch_duration = time.time() - start_time
        print(f"周期 {epoch+1} 总结 (用时: {epoch_duration:.2f}秒):")
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"  验证损失: {val_loss:.4f}, 验证准确率 (TTA): {val_acc:.4f}")
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  当前学习率: {current_lr:.8f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # *** 关键改动：更新保存的模型文件名 ***
            torch.save(model.state_dict(), "DINO_gender_wav_w.pth")
            print(f"  --> 已保存新的最佳模型！准确率: {best_acc:.4f}")
            patience = 0
        else:
            patience += 1
            print(f"  验证集准确率未提升。耐心值: {patience}/{MAX_PATIENCE}")
            if patience >= MAX_PATIENCE:
                print(f"触发早停机制。")
                break
        
        # *** 关键改动：更新保存的曲线图文件名 ***
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, filename="training_curves_gender_amp.png")

    print("\n训练完成！")
    print(f"最终最佳验证准确率 (TTA): {best_acc:.4f}")

if __name__ == "__main__":
    main()














#加速，待跑
# import os
# import time
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoModel
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Subset, Dataset
# from torch.optim import AdamW
# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import pywt
# from PIL import Image
# import warnings
# from torch.cuda.amp import autocast, GradScaler

# warnings.filterwarnings("ignore")

# # =============================================================
# # ==                     类定义 (CLASS DEFINITIONS)            ==
# # =============================================================

# # DINOv2AttentionWaveletClassifier 和 FocalLoss 类保持不变
# class DINOv2AttentionWaveletClassifier(nn.Module):
#     def __init__(self, num_classes, model_path="facebook/dinov2-base"):
#         super().__init__(); self.backbone = AutoModel.from_pretrained(model_path); self.hidden_dim = self.backbone.config.hidden_size; self.fusion_weight = nn.Parameter(torch.zeros(1)); self.classifier = nn.Sequential(nn.Linear(self.hidden_dim * 2, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, num_classes))
#     def forward(self, s, w):
#         bs = s.shape[0]; ci = torch.cat([s, w], dim=0); co = self.backbone(pixel_values=ci); cc = co.last_hidden_state[:, 0]; sc, wc = cc[:bs], cc[bs:]; weight = torch.sigmoid(self.fusion_weight); f = torch.cat([weight * sc, (1 - weight) * wc], dim=-1); return self.classifier(f)
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
#         super().__init__(); self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
#     def forward(self, i, t):
#         ce = F.cross_entropy(i, t, reduction='none'); pt = torch.exp(-ce); loss = (1 - pt) ** self.gamma * ce
#         if self.alpha is not None: loss *= self.alpha[t]
#         return loss.mean() if self.reduction == "mean" else loss.sum()

# # =========================================================================
# # === 关键改动：全新的“智能缓存”数据集类 ===
# # =========================================================================
# class CustomSmartCacheDataset(Dataset):
#     """
#     智能缓存数据集类：
#     - 如果缓存存在，则加载。
#     - 如果缓存不存在，则实时计算，并保存为缓存以供下次使用。
#     """
#     def __init__(self, image_folder_dataset, cache_dir, spatial_transform, normalize_transform, wavelet='db4'):
#         self.image_folder_dataset = image_folder_dataset
#         self.cache_dir = cache_dir
#         self.spatial_transform = spatial_transform
#         self.normalize = normalize_transform
#         self.wavelet = wavelet
#         self.resizer = transforms.Resize((224, 224), antialias=True)
#         # 确保缓存根目录存在
#         os.makedirs(self.cache_dir, exist_ok=True)

#     def _compute_wavelet(self, img_tensor):
#         # 这是小波变换的核心计算逻辑
#         gray = transforms.functional.rgb_to_grayscale(img_tensor, num_output_channels=1)
#         gray_np = gray.squeeze(0).cpu().numpy()
#         coeffs = pywt.dwt2(gray_np, self.wavelet)
#         cA, (cH, cV, cD) = coeffs
#         (h, w) = cA.shape
#         cH, cV, cD = cH[:h, :w], cV[:h, :w], cD[:h, :w]
#         cA = (cA - cA.min()) / (cA.max() - cA.min() + 1e-6)
#         cH = (cH - cH.min()) / (cH.max() - cH.min() + 1e-6)
#         cV = (cV - cV.min()) / (cV.max() - cV.min() + 1e-6)
#         cA_tensor = torch.from_numpy(cA).float().unsqueeze(0)
#         cH_tensor = torch.from_numpy(cH).float().unsqueeze(0)
#         cV_tensor = torch.from_numpy(cV).float().unsqueeze(0)
#         combined_tensor = torch.cat([cA_tensor, cH_tensor, cV_tensor], dim=0)
#         return self.resizer(combined_tensor.unsqueeze(0)).squeeze(0)

#     def __getitem__(self, index):
#         img_path, label = self.image_folder_dataset.imgs[index]
        
#         # 1. 处理空间域图像
#         try:
#             img_pil = Image.open(img_path).convert("RGB")
#             # 空间变换（包括转Tensor和数据增强）
#             spatial_tensor = self.spatial_transform(img_pil)
#         except Exception as e:
#             print(f"打开或处理空间图像 {img_path} 时出错: {e}")
#             return (torch.zeros((3, 224, 224)), torch.zeros((3, 224, 224))), label

#         # 2. 处理小波域图像（智能缓存逻辑）
#         class_name = self.image_folder_dataset.classes[label]
#         cache_sub_dir = os.path.join(self.cache_dir, class_name)
#         cache_filename = os.path.basename(img_path) + ".pt"
#         cache_path = os.path.join(cache_sub_dir, cache_filename)

#         try:
#             if os.path.exists(cache_path):
#                 # 如果缓存存在，直接加载
#                 freq_tensor = torch.load(cache_path).float()
#             else:
#                 # 如果缓存不存在，进行计算
#                 freq_tensor = self._compute_wavelet(spatial_tensor)
#                 # 并保存到缓存目录以备后用（使用半精度以节省空间）
#                 os.makedirs(cache_sub_dir, exist_ok=True)
#                 torch.save(freq_tensor.half(), cache_path)
#         except Exception as e:
#             print(f"处理或加载小波缓存 {cache_path} 时出错: {e}")
#             freq_tensor = torch.zeros((3, 224, 224))

#         # 3. 对两个张量进行归一化
#         spatial_tensor_normalized = self.normalize(spatial_tensor)
#         freq_tensor_normalized = self.normalize(freq_tensor)
        
#         return (spatial_tensor_normalized, freq_tensor_normalized), label

#     def __len__(self):
#         return len(self.image_folder_dataset)


# # 工具函数 (plot_training_curves, train_one_epoch, evaluate) 保持不变
# def plot_training_curves(tl, vl, ta, va, fn="training_curves_gender_smart_cache.png"):
#     e=range(1, len(tl) + 1); plt.figure(figsize=(14, 6)); plt.subplot(1, 2, 1); plt.plot(e, tl, 'bo-', label='训练损失'); plt.plot(e, vl, 'ro-', label='验证损失'); plt.xlabel('周期'); plt.ylabel('损失'); plt.title('损失曲线'); plt.legend(); plt.grid(True); plt.subplot(1, 2, 2); plt.plot(e, ta, 'bo-', label='训练准确率'); plt.plot(e, va, 'ro-', label='验证准确率'); plt.xlabel('周期'); plt.ylabel('准确率'); plt.title('准确率曲线'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(fn); plt.close()
# def train_one_epoch(m, l, o, c, d, s):
#     m.train(); tl, cr = 0, 0
#     for (sx, fx), y in tqdm(l, desc="正在训练 (Smart Cache)"):
#         sx, fx, y = sx.to(d), fx.to(d), y.to(d); o.zero_grad()
#         with autocast(): out = m(sx, fx); loss = c(out, y)
#         if torch.isnan(loss): continue
#         s.scale(loss).backward(); s.step(o); s.update(); tl += loss.item(); cr += (out.argmax(1) == y).sum().item()
#     return tl / len(l), cr / len(l.dataset)
# @torch.no_grad()
# def evaluate(m, l, c, d, cn):
#     m.eval(); tl, cr = 0, 0; ap, al = [], []
#     for (sx, fx), y in tqdm(l, desc="正在评估 (Evaluating with TTA)"):
#         sx, fx, y = sx.to(d), fx.to(d), y.to(d)
#         with autocast(): o1 = m(sx, fx); sxf = transforms.functional.hflip(sx); o2 = m(sxf, fx); oa = (o1 + o2) / 2.0
#         loss = c(oa, y); tl += loss.item(); pred = oa.argmax(1); cr += (pred == y).sum().item(); ap.append(pred.cpu()); al.append(y.cpu())
#     ap, al = torch.cat(ap), torch.cat(al)
#     print("\n分类报告 (TTA):"); print(classification_report(al, ap, target_names=cn, digits=4, zero_division=0))
#     return tl / len(l), cr / len(l.dataset)

# def main():
#     # --- 0. 配置信息 ---
#     DATA_DIR = r"/autodl-tmp/newgender"
#     CACHE_DIR = r"/autodl-tmp/newgender_smart_cache" # 智能缓存目录
#     MODEL_PATH = "facebook/dinov2-base"
#     NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MAX_PATIENCE = 50, 64, 2e-5, 2e-2, 10
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"正在使用设备: {device}")

#     train_spatial_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandAugment(), transforms.ToTensor()])
#     val_spatial_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
#     normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     print("正在加载数据集...")
#     full_dataset_raw = datasets.ImageFolder(DATA_DIR)
#     class_names, num_classes = list(full_dataset_raw.class_to_idx.keys()), len(full_dataset_raw.classes)
#     print(f"找到 {num_classes} 个类别: {class_names}")

#     # 使用新的 CustomSmartCacheDataset
#     train_full_dataset = CustomSmartCacheDataset(full_dataset_raw, CACHE_DIR, train_spatial_transform, normalize_transform)
#     val_full_dataset = CustomSmartCacheDataset(full_dataset_raw, CACHE_DIR, val_spatial_transform, normalize_transform)
#     print(f"智能缓存模式已启用。缓存将保存至: {CACHE_DIR}")
    
#     print("正在对数据集进行分层分割...")
#     labels = np.array(full_dataset_raw.targets)
#     train_idx, val_idx = [], []
#     for c_label in range(num_classes):
#         c_indices = np.where(labels == c_label)[0]; np.random.shuffle(c_indices); v_size = int(len(c_indices) * 0.2)
#         val_idx.extend(c_indices[:v_size]); train_idx.extend(c_indices[v_size:])
#     np.random.shuffle(train_idx); np.random.shuffle(val_idx)
#     print(f"数据集分割完成。训练集: {len(train_idx)}, 验证集: {len(val_idx)}")
#     train_set, val_set = Subset(train_full_dataset, train_idx), Subset(val_full_dataset, val_idx)
    
#     num_workers_to_use = 16 
#     print(f"使用 {num_workers_to_use} 个 workers 进行数据加载。")
#     train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers_to_use, pin_memory=True)
#     val_loader = DataLoader(val_set, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=num_workers_to_use, pin_memory=True)

#     model = DINOv2AttentionWaveletClassifier(num_classes=num_classes, model_path=MODEL_PATH).to(device)
#     if torch.__version__.startswith('2'): print("Compiling model..."); model = torch.compile(model)
#     print("模型初始化完成。")
    
#     train_labels = labels[train_idx]; class_counts = np.bincount(train_labels, minlength=num_classes)
#     class_weights = torch.tensor([len(train_labels) / (c + 1e-6) for c in class_counts], dtype=torch.float32).to(device)
#     criterion = FocalLoss(alpha=class_weights, gamma=2.0)
#     optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True, min_lr=1e-7)
#     scaler = GradScaler()
    
#     best_acc, patience = 0.0, 0
#     print(f"开始智能缓存训练，共 {NUM_EPOCHS} 个周期...")
#     train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
#     for epoch in range(NUM_EPOCHS):
#         start_time = time.time()
#         print(f"\n--- 周期 {epoch+1}/{NUM_EPOCHS} ---")
#         if epoch == 0:
#             print("第一轮训练较慢，正在生成缓存文件...")
#         elif epoch == 1:
#             print("第二轮训练开始，将加载缓存，速度会显著提升！")

#         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
#         val_loss, val_acc = evaluate(model, val_loader, criterion, device, class_names)
        
#         train_losses.append(train_loss); val_losses.append(val_loss); train_accs.append(train_acc); val_accs.append(val_acc)
#         epoch_duration = time.time() - start_time
#         print(f"周期 {epoch+1} 总结 (用时: {epoch_duration:.2f}秒):")
#         print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
#         print(f"  验证损失: {val_loss:.4f}, 验证准确率 (TTA): {val_acc:.4f}")
        
#         scheduler.step(val_acc); current_lr = optimizer.param_groups[0]['lr']; print(f"  当前学习率: {current_lr:.8f}")
        
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save(model.state_dict(), "best_gender_classifier_smart_cache.pth")
#             print(f"  --> 已保存新的最佳模型！准确率: {best_acc:.4f}")
#             patience = 0
#         else:
#             patience += 1
#             print(f"  验证集准确率未提升。耐心值: {patience}/{MAX_PATIENCE}")
#             if patience >= MAX_PATIENCE:
#                 print(f"触发早停机制。")
#                 break
                
#         plot_training_curves(train_losses, val_losses, train_accs, val_accs, filename="training_curves_gender_smart_cache.png")

#     print("\n训练完成！")
#     print(f"最终最佳验证准确率 (TTA): {best_acc:.4f}")

# if __name__ == "__main__":
#     main()