import os
import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc , accuracy_score
from prettytable import PrettyTable
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd
from sklearn.utils import resample
import seaborn as sns
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 0.8,      # 细边框
    'lines.linewidth': 1.1,     # 细线
    'axes.grid': False,         # 关闭背景grid
    'savefig.transparent': True # 导出透明背景
})
sns.set_style('white')
custom_palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
sns.set_palette(custom_palette)
from torch.utils.data import TensorDataset
import json
from sklearn.metrics import precision_score, recall_score

# ========== 1. Utility Functions ==========
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def evaluate_dynamic_fusion(fusion_model, test_logits, test_labels, device, config):
    """评估Dynamic Weight Fusion模型"""

    fusion_model.eval()
    test_dataset = TensorDataset(test_logits, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for logits_batch, labels_batch in test_loader:
            logits_batch = logits_batch.to(device)

            fused_output, weights = fusion_model(logits_batch)
            probs = F.softmax(fused_output, dim=1)
            preds = fused_output.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = calculate_unified_metrics(all_labels, all_preds, all_probs, config['class_names'])

    # 打印结果
    print(f"\n{'=' * 20} Dynamic Fusion测试结果 {'=' * 20}")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"F1分数 (Macro): {metrics['f1_macro']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")

    return {
        'metrics': metrics,
        'predictions': (all_labels, all_preds, all_probs)
    }


def bootstrap_metric(y_true, y_pred, y_prob, metric_func, n_iterations=500, ci=0.95):
    scores = []
    n_samples = len(y_true)

    for _ in range(n_iterations):
        indices = resample(range(n_samples), replace=True, n_samples=n_samples)
        score = metric_func(
            y_true[indices],
            y_pred[indices],
            y_prob[indices] if y_prob is not None else None
        )
        scores.append(score)

    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1 - alpha))
    return np.mean(scores), (lower, upper)

def extract_serial_number(folder_name):
    """Extract serial number from folder name"""
    import re
    match = re.match(r'^(\d+)', folder_name)
    if match:
        return int(match.group(1))
    return None


def extract_yellow_metrics(eye_patch):
    """Extract jaundice-specific color features from eye patches"""
    img = np.array(eye_patch)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Yellow detection in multiple color spaces
    yellow_hsv = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([40, 255, 255]))

    # Extract b* channel from LAB (yellow-blue axis)
    b_channel = lab[:, :, 2]
    yellow_lab = b_channel > 128  # High b* values indicate yellow

    # Combine metrics
    features = [
        np.mean(yellow_hsv > 0),  # Yellow percentage in HSV
        np.mean(yellow_lab),  # Yellow percentage in LAB
        np.std(b_channel),  # Variation in yellowness
        np.percentile(b_channel, 90)  # Top 10% yellowness
    ]

    return np.array(features)


def plot_multiclass_roc(y_true, y_score, class_names, save_path=None):
    import seaborn as sns
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(figsize=(7, 6))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC={roc_auc[i]:.2f})', linewidth=1.1)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right", frameon=False)
    plt.grid(False)
    plt.tight_layout()
    if save_path:
        if save_path.endswith('.png'):
            save_path = save_path.replace('.png', '.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()



def compute_class_weights(labels):
    """Calculate aggressive class weights to handle class imbalance"""
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    # Use square root scaling
    weights = total_samples / (len(class_counts) * np.sqrt(class_counts))
    return torch.FloatTensor(weights)


def color_sensitive_augmentation(image):
    """Apply color-sensitive augmentation for jaundice detection"""
    # Convert to HSV space for enhancement
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Split channels
    h, s, v = cv2.split(hsv_img)

    # Create mask for yellow regions
    yellow_mask = ((h >= 20) & (h <= 60) & (s >= 0.4 * 255) & (v >= 0.4 * 255))

    # Apply random adjustments to non-yellow regions
    non_yellow_mask = ~yellow_mask
    h_shift = np.random.uniform(-5, 5)  # Slight hue adjustment
    s_scale = np.random.uniform(0.8, 1.2)  # Saturation adjustment
    v_scale = np.random.uniform(0.8, 1.2)  # Brightness adjustment

    h[non_yellow_mask] = np.clip(h[non_yellow_mask] + h_shift, 0, 179)
    s[non_yellow_mask] = np.clip(s[non_yellow_mask] * s_scale, 0, 255)
    v[non_yellow_mask] = np.clip(v[non_yellow_mask] * v_scale, 0, 255)

    # Merge channels and convert back to RGB
    hsv_img = cv2.merge([h, s, v])
    rgb_img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return rgb_img


# ========== 2. Face Feature Extraction ==========

class Jaundice3ClassDataset_ImageOnly(Dataset):
    """只使用图像的数据集类（用于图像模型训练）"""

    def __init__(self, root_dir, patient_names, split='train',
                 bilirubin_csv=None, transform=None, target_size=(224, 224),
                 enable_undersample=False, undersample_count=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.label_names = ['mild', 'moderate', 'severe']
        self.class_to_idx = {name: i for i, name in enumerate(self.label_names)}
        self.samples = []

        # 读取胆红素数据
        assert bilirubin_csv is not None, "Must provide bilirubin csv file path"
        df = pd.read_excel(bilirubin_csv)
        df.columns = df.columns.str.strip()
        self.serial2bil = dict(zip(df['序号'], df['26、总胆红素值（umol/L）']))

        for folder_name in patient_names:
            serial = extract_serial_number(folder_name)
            if serial is None or serial not in self.serial2bil:
                print(f"[Warning] {folder_name} not matched to table serial number, skipping")
                continue

            bil = float(self.serial2bil[serial])
            # 直接三分类
            if bil < 171:
                label_name = 'mild'
            elif bil < 342:
                label_name = 'moderate'
            else:
                label_name = 'severe'
            label = self.class_to_idx[label_name]

            patient_folder = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(patient_folder):
                continue

            imgs = [f for f in os.listdir(patient_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not imgs:
                print(f"[Warning] No images in {patient_folder}, skipping")
                continue

            for img_name in imgs:
                img_path = os.path.join(patient_folder, img_name)
                self.samples.append((img_path, label, serial))

        # 下采样处理
        if self.split == 'train' and enable_undersample:
            print("Applying undersampling to balance classes...")
            label_to_samples = defaultdict(list)
            for sample in self.samples:
                label_to_samples[sample[1]].append(sample)
            if undersample_count is None:
                undersample_count = min(len(s) for s in label_to_samples.values())
            print(f"Undersample count per class: {undersample_count}")
            new_samples = []
            for label, group in label_to_samples.items():
                if len(group) >= undersample_count:
                    new_samples.extend(random.sample(group, undersample_count))
                else:
                    new_samples.extend(group)
            self.samples = new_samples
            print(f"After undersampling: {Counter([s[1] for s in self.samples])}")

        print(f"Loaded {len(self.samples)} samples. Class distribution:")
        print(Counter([s[1] for s in self.samples]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_path, label, serial = self.samples[idx]
            img = Image.open(img_path).convert('RGB')

            # Apply data augmentation for training
            if self.split == 'train':
                np_img = np.array(img)
                np_img = color_sensitive_augmentation(np_img)
                img = Image.fromarray(np_img)

            # Apply transform if provided
            if self.transform:
                img_tensor = self.transform(img)
            else:
                # Default transform
                transform = T.Compose([
                    T.Resize(self.target_size),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = transform(img)

            return img_tensor, torch.tensor(label), serial

        except Exception as e:
            print(f"Cannot process sample {idx}, error: {e}")
            return self.__getitem__(max(0, idx - 1))


class YellowFeatureDataset(Dataset):
    """只包含黄疸特征的数据集"""

    def __init__(self, root_dir, patient_names, bilirubin_csv=None,
                 enable_undersample=False, undersample_count=None):
        self.root_dir = root_dir
        self.ffe = FacialFeatureExtractor(use_mediapipe=True)

        # 读取胆红素标签
        df = pd.read_excel(bilirubin_csv)
        self.serial2bil = dict(zip(df['序号'], df['26、总胆红素值（umol/L）']))

        self.samples = []
        for folder_name in patient_names:
            serial = extract_serial_number(folder_name)
            if serial is None or serial not in self.serial2bil:
                continue

            bil = float(self.serial2bil[serial])
            # 三分类标签
            if bil < 171:
                label = 0  # mild
            elif bil < 342:
                label = 1  # moderate
            else:
                label = 2  # severe

            # 收集该患者的所有图像
            patient_folder = os.path.join(self.root_dir, folder_name)
            if not os.path.exists(patient_folder):
                continue

            imgs = [f for f in os.listdir(patient_folder)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

            for img_name in imgs:
                img_path = os.path.join(patient_folder, img_name)
                self.samples.append((img_path, label, serial))

        print(f"Loaded {len(self.samples)} yellow feature samples. Class distribution:")
        labels = [sample[1] for sample in self.samples]
        print(Counter(labels))

        # 下采样处理
        if enable_undersample and undersample_count:
            print("Applying undersampling to yellow features...")
            self.samples = self._apply_undersampling(self.samples, undersample_count)

    def _apply_undersampling(self, samples, undersample_count):
        """对黄疸特征数据进行下采样"""
        print(f"Undersample count per class: {undersample_count}")

        # 按类别分组
        class_samples = {0: [], 1: [], 2: []}
        for sample in samples:
            class_samples[sample[1]].append(sample)

        # 对每个类别进行下采样
        undersampled = []
        for class_id in [0, 1, 2]:
            class_data = class_samples[class_id]
            if len(class_data) > undersample_count:
                # 随机选择
                selected = random.sample(class_data, undersample_count)
                undersampled.extend(selected)
            else:
                # 如果样本不足，全部保留
                undersampled.extend(class_data)

        # 打印下采样后的分布
        labels = [sample[1] for sample in undersampled]
        print(f"After undersampling: {Counter(labels)}")

        return undersampled

    def extract_yellow_features(self, image):
        """从图像中提取8维黄疸特征"""
        try:
            # 提取左右眼区域
            left_eye, right_eye = self.ffe.extract_sclera_patches(image)

            # 从每个眼部区域提取4个特征
            left_features = extract_yellow_metrics(left_eye)  # 4维
            right_features = extract_yellow_metrics(right_eye)  # 4维

            # 合并为8维特征向量
            features = np.concatenate([left_features, right_features])

            # 检查特征是否有效
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                print("Warning: Invalid features detected, using zeros")
                return np.zeros(8)

            return features

        except Exception as e:
            print(f"特征提取失败: {e}")
            return np.zeros(8)  # 返回零向量作为默认值

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, serial = self.samples[idx]

        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')

            # 提取黄疸特征
            yellow_features = self.extract_yellow_features(image)

            return {
                'yellow_features': torch.FloatTensor(yellow_features),
                'label': label,  # 直接返回int，不转换为tensor
                'serial': serial,
                'img_path': img_path
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # 返回默认值
            return {
                'yellow_features': torch.zeros(8),
                'label': 0,
                'serial': -1,
                'img_path': img_path
            }


class YellowFeatureClassifier(nn.Module):
    """专门用于黄疸特征分类的网络"""

    def __init__(self, input_dim=8, hidden_dim=64, num_classes=3, dropout=0.3):
        super(YellowFeatureClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class Jaundice3ClassDataset_Full(Dataset):
    """包含图像和黄疸特征的完整数据集类（用于Dynamic Weight Fusion训练）"""

    def __init__(self, root_dir, patient_names, split='train',
                 bilirubin_csv=None, transform=None,
                 enable_undersample=False, undersample_count=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_names = ['mild', 'moderate', 'severe']
        self.class_to_idx = {name: i for i, name in enumerate(self.label_names)}

        # 初始化特征提取器
        self.ffe = FacialFeatureExtractor(use_mediapipe=True)

        self.samples = []

        # 读取胆红素数据
        assert bilirubin_csv is not None, "Must provide bilirubin csv file path"
        df = pd.read_excel(bilirubin_csv)
        df.columns = df.columns.str.strip()
        self.serial2bil = dict(zip(df['序号'], df['26、总胆红素值（umol/L）']))

        for folder_name in patient_names:
            serial = extract_serial_number(folder_name)
            if serial is None or serial not in self.serial2bil:
                print(f"[Warning] {folder_name} not matched to table serial number, skipping")
                continue

            bil = float(self.serial2bil[serial])
            # 直接三分类
            if bil < 171:
                label_name = 'mild'
            elif bil < 342:
                label_name = 'moderate'
            else:
                label_name = 'severe'
            label = self.class_to_idx[label_name]

            patient_folder = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(patient_folder):
                continue

            imgs = [f for f in os.listdir(patient_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not imgs:
                print(f"[Warning] No images in {patient_folder}, skipping")
                continue

            for img_name in imgs:
                img_path = os.path.join(patient_folder, img_name)
                self.samples.append((img_path, label, serial))

        # 下采样处理
        if self.split == 'train' and enable_undersample:
            print("Applying undersampling to full dataset...")
            label_to_samples = defaultdict(list)
            for sample in self.samples:
                label_to_samples[sample[1]].append(sample)
            if undersample_count is None:
                undersample_count = min(len(s) for s in label_to_samples.values())
            print(f"Undersample count per class: {undersample_count}")
            new_samples = []
            for label, group in label_to_samples.items():
                if len(group) >= undersample_count:
                    new_samples.extend(random.sample(group, undersample_count))
                else:
                    new_samples.extend(group)
            self.samples = new_samples
            print(f"After undersampling: {Counter([s[1] for s in self.samples])}")

        print(f"Loaded {len(self.samples)} full samples. Class distribution:")
        print(Counter([s[1] for s in self.samples]))

    def extract_yellow_features(self, image):
        """从PIL图像中提取黄疸特征"""
        try:
            # 提取眼部区域
            left_eye, right_eye = self.ffe.extract_sclera_patches(image)

            # 确保区域提取成功
            if left_eye is None or right_eye is None:
                # 回退到直接裁剪
                w, h = image.size
                left_eye = image.crop((0, 0, w // 4, h // 4)) if left_eye is None else left_eye
                right_eye = image.crop((3 * w // 4, 0, w, h // 4)) if right_eye is None else right_eye

            # 提取黄疸指标
            left_eye_yellow = extract_yellow_metrics(left_eye)
            right_eye_yellow = extract_yellow_metrics(right_eye)

            # 合并特征 (每个眼部4个特征，总共8个)
            features = np.concatenate([left_eye_yellow, right_eye_yellow])
            return features

        except Exception as e:
            print(f"特征提取错误: {e}")
            return np.zeros(8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_path, label, serial = self.samples[idx]
            img = Image.open(img_path).convert('RGB')

            # Apply data augmentation for training
            if self.split == 'train':
                np_img = np.array(img)
                np_img = color_sensitive_augmentation(np_img)
                img = Image.fromarray(np_img)

            # 提取黄疸特征（在变换之前）
            yellow_features = self.extract_yellow_features(img)

            # Apply transform if provided
            if self.transform:
                img_tensor = self.transform(img)
            else:
                # Default transform
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = transform(img)

            return {
                'image': img_tensor,
                'yellow_features': torch.FloatTensor(yellow_features),
                'label': torch.tensor(label),
                'serial': serial
            }

        except Exception as e:
            print(f"Cannot process sample {idx}, error: {e}")
            return self.__getitem__(max(0, idx - 1))

def train_yellow_classifier(train_loader, val_loader, device, config):
    """训练黄疸特征分类器"""

    # 创建模型
    model = YellowFeatureClassifier(
        input_dim=8,  # 8维黄疸特征
        hidden_dim=config['yellow_classifier']['hidden_dim'],
        num_classes=3,
        dropout=config['yellow_classifier']['dropout']
    ).to(device)

    # 计算类别权重（处理数据不平衡）
    print("计算类别权重...")
    all_labels = []
    for batch in train_loader:
        # 修复：正确提取标签
        labels = batch['label']
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.cpu().numpy().tolist())
        elif isinstance(labels, list):
            all_labels.extend(labels)
        else:
            all_labels.append(int(labels))

    # 转换为numpy数组
    all_labels = np.array(all_labels, dtype=int)
    print(f"训练集标签分布: {np.bincount(all_labels)}")

    # 计算类别权重
    class_weights = compute_class_weights(all_labels).to(device)
    print(f"类别权重: {class_weights}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['yellow_classifier']['lr'],
        weight_decay=config['yellow_classifier']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    best_f1 = 0
    patience_counter = 0

    print(f"开始训练黄疸特征分类器，共 {config['yellow_classifier']['epochs']} 轮...")

    for epoch in range(config['yellow_classifier']['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            features = batch['yellow_features'].to(device)  # [batch_size, 8]
            labels = batch['label']

            # 确保labels是tensor格式
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)  # [batch_size, 3]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        # 验证阶段
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['yellow_features'].to(device)
                labels = batch['label']

                # 确保labels是tensor格式
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # 计算指标
        train_acc = train_correct / train_total
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        scheduler.step(val_f1)

        # 每10轮打印一次详细信息
        if epoch % 10 == 0 or epoch == config['yellow_classifier']['epochs'] - 1:
            print(f"Epoch {epoch + 1}/{config['yellow_classifier']['epochs']}: "
                  f"Train Loss: {train_loss / len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"Val F1: {val_f1:.4f}")

        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            model_path = os.path.join(config['output_dir'], 'best_yellow_classifier.pth')
            torch.save(model.state_dict(), model_path)
            print(f"✓ 新的最佳模型已保存 (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 20:  # 早停
                print(f"早停触发，最佳F1: {best_f1:.4f}")
                break

    # 加载最佳模型
    print("加载最佳黄疸特征分类器...")
    model_path = os.path.join(config['output_dir'], 'best_yellow_classifier.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    print(f"✓ 黄疸特征分类器训练完成，最佳F1: {best_f1:.4f}")
    return model


def train_ensemble_weights(image_models, yellow_model, train_loader, val_loader, device, config):
    """训练集成权重 - 使用LayeredEnsemble的简单版本"""

    ensemble_model = LayeredEnsemble(
        num_models=5,
        num_classes=3,
        fusion_method='weighted'
    ).to(device)

    optimizer = torch.optim.AdamW(ensemble_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    for epoch in range(20):  # 简单训练
        ensemble_model.train()
        for batch in train_loader:
            images = batch['image'].to(device)
            yellow_features = batch['yellow_features'].to(device)
            labels = batch['label'].to(device)

            # 获取所有模型logits
            logits_list = []
            for model in image_models.values():
                _, logits = model(images)
                logits_list.append(logits)

            yellow_logits = yellow_model(yellow_features)
            logits_list.append(yellow_logits)

            # 集成预测
            ensemble_output = ensemble_model(logits_list)
            loss = criterion(ensemble_output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Ensemble Epoch {epoch + 1}/20 completed")

    return ensemble_model


class LayeredEnsemble(nn.Module):
    """分层集成模型"""

    def __init__(self, num_models=5, num_classes=3, fusion_method='weighted'):
        super(LayeredEnsemble, self).__init__()
        self.num_models = num_models  # 4个图像模型 + 1个黄疸特征模型
        self.num_classes = num_classes
        self.fusion_method = fusion_method

        if fusion_method == 'weighted':
            # 学习权重参数
            self.weights = nn.Parameter(torch.ones(num_models) / num_models)

    def forward(self, logits_list):
        """
        logits_list: list of tensors, each tensor is [batch_size, num_classes]
        """
        # 堆叠所有logits
        stacked_logits = torch.stack(logits_list, dim=1)  # [batch_size, num_models, num_classes]

        if self.fusion_method == 'weighted':
            # 加权平均
            weights = torch.softmax(self.weights, dim=0)
            weighted_logits = torch.sum(stacked_logits * weights.view(1, -1, 1), dim=1)
            return weighted_logits

        elif self.fusion_method == 'voting':
            # 软投票
            averaged_logits = torch.mean(stacked_logits, dim=1)
            return averaged_logits

class DynamicWeightFusion(nn.Module):
    def __init__(self, num_models=5, num_classes=3, hidden_dim=64):
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes

        # 权重生成网络 - 根据输入动态生成每个模型的权重
        self.weight_generator = nn.Sequential(
            nn.Linear(num_models * num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, num_models),
            nn.Softmax(dim=-1)  # 确保权重和为1
        )

        # 可选：添加温度参数来控制权重的锐度
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, logits):
        """
        Args:
            logits: [batch_size, num_models, num_classes] - 5个模型的logits输出
        Returns:
            weighted_logits: [batch_size, num_classes] - 融合后的输出
            weights: [batch_size, num_models] - 每个样本的模型权重
        """
        batch_size = logits.size(0)

        # 将所有模型的logits展平作为权重生成器的输入
        flattened = logits.view(batch_size, -1)  # [batch_size, num_models * num_classes]

        # 动态生成权重
        weights = self.weight_generator(flattened)  # [batch_size, num_models]

        # 应用温度缩放
        weights = F.softmax(weights / self.temperature, dim=-1)

        # 加权融合
        weighted_logits = torch.sum(
            logits * weights.unsqueeze(-1), dim=1
        )  # [batch_size, num_classes]

        return weighted_logits, weights
class FacialFeatureExtractor:
    def __init__(self, use_mediapipe=True):
        self.use_mediapipe = use_mediapipe
        if use_mediapipe:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

        # Define facial landmark indices
        self.left_eye_indices = list(range(362, 374)) + list(range(263, 273))
        self.right_eye_indices = list(range(133, 145)) + list(range(33, 43))

    def get_face_landmarks(self, image):
        if not self.use_mediapipe:
            return None, (image.height, image.width, 3) if isinstance(image, Image.Image) else image.shape

        if isinstance(image, Image.Image):
            image = np.array(image)
        try:
            results = self.face_mesh.process(image[..., ::-1])
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark, image.shape
        except Exception as e:
            print(f"Warning: Face landmark detection failed: {e}")
        return None, image.shape

    def extract_sclera_patches(self, image, expand_ratio=1.5):
        """Extract left and right eye patches from the image"""
        landmarks, img_shape = self.get_face_landmarks(image)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if landmarks is None:
            # Fallback to basic cropping if landmarks not detected
            w, h = image.size
            size = min(w, h) // 6
            left_eye = image.crop((w // 4 - size, h // 3 - size, w // 4 + size, h // 3 + size))
            right_eye = image.crop((3 * w // 4 - size, h // 3 - size, 3 * w // 4 + size, h // 3 + size))
            return left_eye, right_eye

        height, width = img_shape[:2]

        def crop_eye(indices):
            pts = [(landmarks[idx].x * width, landmarks[idx].y * height) for idx in indices]
            min_x, max_x = min(p[0] for p in pts), max(p[0] for p in pts)
            min_y, max_y = min(p[1] for p in pts), max(p[1] for p in pts)
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            w, h = (max_x - min_x) * expand_ratio, (max_y - min_y) * expand_ratio
            return image.crop((max(0, int(cx - w / 2)), max(0, int(cy - h / 2)),
                               min(width, int(cx + w / 2)), min(height, int(cy + h / 2))))

        return crop_eye(self.left_eye_indices), crop_eye(self.right_eye_indices)


# ========== 3. Dataset Class ==========
class Jaundice3ClassDataset(Dataset):
    def __init__(self, root_dir, patient_names, split='train',
                 bilirubin_csv=None, transform=None, use_mediapipe=True,
                 include_yellow_features=True, enable_undersample=False, undersample_count=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.include_yellow_features = include_yellow_features
        self.label_names = ['mild', 'moderate', 'severe']
        self.class_to_idx = {name: i for i, name in enumerate(self.label_names)}
        self.ffe = FacialFeatureExtractor(use_mediapipe=use_mediapipe) if include_yellow_features else None
        self.samples = []

        # 读取胆红素数据
        assert bilirubin_csv is not None, "Must provide bilirubin csv file path"
        df = pd.read_excel(bilirubin_csv)
        df.columns = df.columns.str.strip()
        self.serial2bil = dict(zip(df['序号'], df['26、总胆红素值（umol/L）']))

        for folder_name in patient_names:
            serial = extract_serial_number(folder_name)
            if serial is None or serial not in self.serial2bil:
                print(f"[Warning] {folder_name} not matched to table serial number, skipping")
                continue

            bil = float(self.serial2bil[serial])
            # 直接三分类
            if bil < 171:
                label_name = 'mild'
            elif bil < 342:
                label_name = 'moderate'
            else:
                label_name = 'severe'
            label = self.class_to_idx[label_name]

            patient_folder = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(patient_folder):
                continue

            imgs = [f for f in os.listdir(patient_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not imgs:
                print(f"[Warning] No images in {patient_folder}, skipping")
                continue

            for img_name in imgs:
                img_path = os.path.join(patient_folder, img_name)
                self.samples.append((img_path, label, serial))
        if self.split == 'train' and enable_undersample:
            print("Applying undersampling to balance classes...")
            label_to_samples = defaultdict(list)
            for sample in self.samples:
                label_to_samples[sample[1]].append(sample)
            if undersample_count is None:
                undersample_count = min(len(s) for s in label_to_samples.values())
            print(f"Undersample count per class: {undersample_count}")
            new_samples = []
            for label, group in label_to_samples.items():
                if len(group) >= undersample_count:
                    new_samples.extend(random.sample(group, undersample_count))
                else:
                    new_samples.extend(group)
            self.samples = new_samples
            print(f"After undersampling: {Counter([s[1] for s in self.samples])}")

        print(f"Loaded {len(self.samples)} samples. Class distribution:")
        print(Counter([s[1] for s in self.samples]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_path, label, serial = self.samples[idx]
            img = Image.open(img_path).convert('RGB')

            # Apply data augmentation for training
            if self.split == 'train':
                np_img = np.array(img)
                np_img = color_sensitive_augmentation(np_img)
                img = Image.fromarray(np_img)

            # Apply transform if provided
            if self.transform:
                img_tensor = self.transform(img)
            else:
                # Default transform
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = transform(img)

            # Extract additional yellow features if needed
            if self.include_yellow_features and self.ffe:
                try:
                    # Extract eye regions
                    left_eye, right_eye = self.ffe.extract_sclera_patches(img)

                    # Ensure regions were extracted successfully
                    if left_eye is None or right_eye is None:
                        # Fallback to direct crops
                        w, h = img.size
                        left_eye = img.crop((0, 0, w // 4, h // 4)) if left_eye is None else left_eye
                        right_eye = img.crop((3 * w // 4, 0, w, h // 4)) if right_eye is None else right_eye

                    # Extract yellow metrics
                    left_eye_yellow = torch.tensor(extract_yellow_metrics(left_eye), dtype=torch.float32)
                    right_eye_yellow = torch.tensor(extract_yellow_metrics(right_eye), dtype=torch.float32)

                    # Combine yellow features
                    eyes_yellow = torch.cat([left_eye_yellow, right_eye_yellow])

                    return img_tensor, eyes_yellow, torch.tensor(label), serial

                except Exception as e:
                    print(f"Feature extraction error, file: {img_path}, error: {e}")
                    # Fall back to previous sample
                    return self.__getitem__(max(0, idx - 1))

            # If no additional features needed, return only the image tensor
            return img_tensor, torch.tensor(label), serial

        except Exception as e:
            print(f"Cannot process sample {idx}, error: {e}")
            # Try returning another sample index
            return self.__getitem__(max(0, idx - 1))  # Return previous valid sample


# ========== 4. SOTA Model Backbones ==========
class SOTABackbone(nn.Module):
    """Base class for SOTA model backbones"""

    def __init__(self, model_name, num_classes=3, pretrained=True, freeze_base=True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = None
        self.feature_dim = 0
        self._build_model(pretrained, freeze_base)

    def _build_model(self, pretrained, freeze_base):
        """Build the model architecture - to be implemented by subclasses"""
        raise NotImplementedError

    def forward(self, x):
        """Forward pass - returns both features and logits"""
        raise NotImplementedError

class ConvNextBackbone(SOTABackbone):
    """ConvNeXt model backbone"""

    def _build_model(self, pretrained, freeze_base):
        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None

        self.backbone = torchvision.models.convnext_base(weights=weights)
        self.feature_dim = 1024

        # Replace classifier with identity to get features
        self.original_classifier = self.backbone.classifier
        self.backbone.classifier = nn.Identity()

        # Create a new classifier
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

        # Freeze base if needed
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)  # 期望 [B, 1024]
        if features.ndim > 2:
            features = torch.flatten(features, 1)  # flatten 所有非batch维
        logits = self.classifier(features)
        return features, logits


class SwinTransformerBackbone(SOTABackbone):
    """Swin Transformer model backbone"""

    def _build_model(self, pretrained, freeze_base):
        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None

        self.backbone = torchvision.models.swin_v2_b(weights=weights)
        self.feature_dim = 1024

        # Store original head
        self.original_head = self.backbone.head

        # Remove classifier head to get features
        self.backbone.head = nn.Identity()

        # Create a new classifier
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

        # Freeze base if needed
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return features, logits


class EfficientNetBackbone(SOTABackbone):
    """EfficientNet model backbone"""

    def _build_model(self, pretrained, freeze_base):
        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None

        self.backbone = torchvision.models.efficientnet_v2_l(weights=weights)
        self.feature_dim = 1280

        # Store original classifier
        self.original_classifier = self.backbone.classifier

        # Remove classifier to get features
        self.backbone.classifier = nn.Identity()

        # Create a new classifier
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

        # Freeze base if needed
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return features, logits


class ViTBackbone(SOTABackbone):
    """Vision Transformer model backbone"""

    def _build_model(self, pretrained, freeze_base):
        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None

        self.backbone = torchvision.models.vit_b_16(weights=weights)
        self.feature_dim = 768

        # Store original head
        self.original_head = self.backbone.heads

        # Remove classifier head to get features
        self.backbone.heads = nn.Identity()

        # Create a new classifier
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

        # Freeze base if needed
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return features, logits


# ========== 5. Ensemble Model ==========
class MLPEnsemble(nn.Module):
    """MLP Ensemble model that combines features from multiple backbones"""

    def __init__(self, feature_dims, yellow_dim=8, num_classes=3, hidden_dims=[512, 256, 128]):
        super().__init__()

        # Calculate total input dimension
        total_dim = sum(feature_dims) + yellow_dim

        # Build MLP layers
        layers = []
        prev_dim = total_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features_list, yellow=None):
        # Concatenate all features
        # Each element in features_list is a tensor of shape [batch_size, feature_dim]
        all_features = torch.cat(features_list, dim=1)

        # Add yellow features if provided
        if yellow is not None:
            all_features = torch.cat([all_features, yellow], dim=1)

        # Forward through MLP
        return self.mlp(all_features)

class MLPEnsembleLogits(nn.Module):
    def __init__(self, num_models, num_classes, yellow_dim=8, hidden_dims=[64, 32]):
        super().__init__()
        input_dim = num_models * num_classes + yellow_dim
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, logits_list, yellow=None):
        x = torch.cat(logits_list, dim=1)
        if yellow is not None:
            x = torch.cat([x, yellow], dim=1)
        return self.mlp(x)

class LogitsFusionEnsemble(nn.Module):
    def __init__(self, num_models, num_classes, hidden_dim=64, yellow_dim=8):
        super(LogitsFusionEnsemble, self).__init__()
        self.fusion_layer = DynamicWeightFusion(
            num_models=num_models,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )

    def forward(self, logits_list, yellow_features=None):
        # 堆叠logits
        stacked_logits = torch.stack(logits_list, dim=1)  # [B, num_models, num_classes]
        fused_logits, weights = self.fusion_layer(stacked_logits)
        return fused_logits, weights



class EarlyStopping:
    """早停机制实现"""

    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print('Restored best weights')
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """保存模型权重"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()




def plot_training_history(history, config):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='#4C72B0')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='#C44E52')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[0, 1].plot(history['train_acc'], label='Train Acc', color='#55A868')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', color='#8172B3')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1曲线
    axes[1, 0].plot(history['train_f1'], label='Train F1', color='#CCB974')
    axes[1, 0].plot(history['val_f1'], label='Val F1', color='#64B5CD')
    axes[1, 0].set_title('F1 Score Curves')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 过拟合检测
    train_val_gap = np.array(history['train_f1']) - np.array(history['val_f1'])
    axes[1, 1].plot(train_val_gap, label='Train-Val F1 Gap', color='#C44E52')
    axes[1, 1].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
    axes[1, 1].set_title('Overfitting Detection')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config['output_dir'], 'training_history.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"训练历史图已保存到: {save_path}")


# ========== 6. Training and Evaluation Functions ==========
def extract_all_logits(data_loader, image_models, yellow_model, device):
    """提取所有模型的logits"""
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="提取logits"):
            images = batch['image'].to(device)
            yellow_features = batch['yellow_features'].to(device)
            labels = batch['label']

            batch_logits = []

            # 图像模型logits
            for model in image_models.values():
                _, logits = model(images)
                batch_logits.append(logits.cpu())

            # 黄疸模型logits
            yellow_logits = yellow_model(yellow_features)
            batch_logits.append(yellow_logits.cpu())

            # 堆叠 [batch_size, num_models, num_classes]
            stacked_logits = torch.stack(batch_logits, dim=1)
            all_logits.append(stacked_logits)
            all_labels.append(labels)

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def train_dynamic_fusion_simple(train_logits, train_labels, val_logits, val_labels, device, config):
    """简化的Dynamic Weight Fusion训练"""

    fusion_model = DynamicWeightFusion(
        num_models=5,  # 4个图像模型 + 1个黄疸模型
        num_classes=3,
        hidden_dim=config['ensemble']['hidden_dim']
    ).to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(train_logits, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(val_logits, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 训练配置
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=config['ensemble']['lr'])
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    patience_counter = 0

    for epoch in range(config['ensemble']['epochs']):
        # 训练
        fusion_model.train()
        for logits_batch, labels_batch in train_loader:
            logits_batch = logits_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            fused_output, _ = fusion_model(logits_batch)
            loss = criterion(fused_output, labels_batch)
            loss.backward()
            optimizer.step()

        # 验证
        fusion_model.eval()
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for logits_batch, labels_batch in val_loader:
                logits_batch = logits_batch.to(device)
                fused_output, _ = fusion_model(logits_batch)
                preds = fused_output.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels_batch.numpy())

        val_f1 = f1_score(val_labels_list, val_preds, average='macro')

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(fusion_model.state_dict(),
                       os.path.join(config['output_dir'], 'best_dynamic_fusion.pth'))
        else:
            patience_counter += 1
            if patience_counter >= config['ensemble']['patience']:
                break

        print(f"Epoch {epoch + 1}: Val F1 = {val_f1:.4f}")

    # 加载最佳模型
    fusion_model.load_state_dict(torch.load(os.path.join(config['output_dir'], 'best_dynamic_fusion.pth')))
    return fusion_model


def plot_comprehensive_roc(unified_results, config):
    """绘制所有模型的综合ROC曲线"""

    plt.figure(figsize=(10, 8))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']

    for i, (model_name, result) in enumerate(unified_results.items()):
        y_true, y_pred, y_prob = result['predictions']

        # 计算每个类别的ROC
        n_classes = len(config['class_names'])
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        # 计算macro-average ROC AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for class_idx in range(n_classes):
            fpr[class_idx], tpr[class_idx], _ = roc_curve(y_true_bin[:, class_idx], y_prob[:, class_idx])
            roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

        # 计算macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[j] for j in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for class_idx in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[class_idx], tpr[class_idx])
        mean_tpr /= n_classes

        macro_auc = auc(all_fpr, mean_tpr)

        plt.plot(all_fpr, mean_tpr,
                 label=f'{model_name} (AUC = {macro_auc:.3f})',
                 color=colors[i % len(colors)], linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comprehensive ROC Curves - All Models')
    plt.legend(loc="lower right", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(config['output_dir'], 'comprehensive_roc_curves.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"综合ROC曲线已保存到: {save_path}")


def plot_comprehensive_prc(unified_results, config):
    """绘制所有模型的综合PRC曲线对比"""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure(figsize=(10, 8))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']

    macro_aps = {}  # 存储每个模型的macro-average AP

    for i, (model_name, result) in enumerate(unified_results.items()):
        y_true, y_pred, y_prob = result['predictions']

        # 计算macro-average PRC
        n_classes = len(config['class_names'])
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        # 计算每个类别的AP然后平均
        class_aps = []
        for class_idx in range(n_classes):
            ap = average_precision_score(y_true_bin[:, class_idx], y_prob[:, class_idx])
            class_aps.append(ap)

        macro_ap = np.mean(class_aps)
        macro_aps[model_name] = macro_ap

        # 计算macro-average precision-recall曲线
        all_y_true = y_true_bin.ravel()
        all_y_scores = y_prob.ravel()
        macro_precision, macro_recall, _ = precision_recall_curve(all_y_true, all_y_scores)

        plt.plot(macro_recall, macro_precision,
                 label=f'{model_name} (mAP = {macro_ap:.3f})',
                 color=colors[i % len(colors)], linewidth=2)

    # 添加基线（随机分类器）
    baseline_ap = np.mean([np.sum(y_true == i) / len(y_true) for i in range(n_classes)])
    plt.axhline(y=baseline_ap, color='gray', linestyle=':',
                label=f'Random Baseline (AP = {baseline_ap:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comprehensive Precision-Recall Curves - All Models')
    plt.legend(loc="lower left", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(config['output_dir'], 'comprehensive_prc_curves.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"综合PRC曲线已保存到: {save_path}")
    return macro_aps


def plot_individual_model_roc(unified_results, config):
    """为每个模型绘制单独的ROC曲线"""

    for model_name, result in unified_results.items():
        y_true, y_pred, y_prob = result['predictions']

        plt.figure(figsize=(8, 6))
        n_classes = len(config['class_names'])
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        colors = ['#4C72B0', '#55A868', '#C44E52']

        # 绘制每个类别的ROC曲线
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr,
                     label=f'{config["class_names"][i]} (AUC = {roc_auc:.3f})',
                     color=colors[i], linewidth=2)

        # 计算并绘制macro-average ROC
        fpr_macro = dict()
        tpr_macro = dict()
        for i in range(n_classes):
            fpr_macro[i], tpr_macro[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])

        all_fpr = np.unique(np.concatenate([fpr_macro[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_macro[i], tpr_macro[i])
        mean_tpr /= n_classes

        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr,
                 label=f'Macro-average (AUC = {macro_auc:.3f})',
                 color='black', linestyle='--', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k:', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curves')
        plt.legend(loc="lower right", frameon=False)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(config['output_dir'], f'{model_name}_roc_curves.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()

        print(f"{model_name} ROC曲线已保存到: {save_path}")


def plot_individual_model_prc(unified_results, config):
    """为每个模型绘制单独的PRC曲线"""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    for model_name, result in unified_results.items():
        y_true, y_pred, y_prob = result['predictions']

        plt.figure(figsize=(8, 6))
        n_classes = len(config['class_names'])
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        colors = ['#4C72B0', '#55A868', '#C44E52']

        # 绘制每个类别的PRC曲线
        class_aps = []
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            class_aps.append(ap)

            plt.plot(recall, precision,
                     label=f'{config["class_names"][i]} (AP = {ap:.3f})',
                     color=colors[i], linewidth=2)

        # 计算并绘制macro-average PRC
        macro_ap = np.mean(class_aps)
        all_y_true = y_true_bin.ravel()
        all_y_scores = y_prob.ravel()
        macro_precision, macro_recall, _ = precision_recall_curve(all_y_true, all_y_scores)

        plt.plot(macro_recall, macro_precision,
                 label=f'Macro-average (AP = {macro_ap:.3f})',
                 color='black', linestyle='--', linewidth=2)

        # 添加基线
        baseline_ap = np.mean([np.sum(y_true == i) / len(y_true) for i in range(n_classes)])
        plt.axhline(y=baseline_ap, color='gray', linestyle=':',
                    label=f'Baseline (AP = {baseline_ap:.3f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} - Precision-Recall Curves')
        plt.legend(loc="lower left", frameon=False)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(config['output_dir'], f'{model_name}_prc_curves.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()

        print(f"{model_name} PRC曲线已保存到: {save_path}")


def plot_confusion_matrices(unified_results, config):
    """绘制所有模型的混淆矩阵对比"""

    n_models = len(unified_results)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (model_name, result) in enumerate(unified_results.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        y_true, y_pred, y_prob = result['predictions']
        cm = confusion_matrix(y_true, y_pred)

        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # 绘制热图
        im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues')

        # 添加文本标注
        thresh = cm_percent.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)',
                        ha="center", va="center",
                        color="white" if cm_percent[i, j] > thresh else "black",
                        fontsize=10)

        ax.set_title(f'{model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_xticks(range(len(config['class_names'])))
        ax.set_yticks(range(len(config['class_names'])))
        ax.set_xticklabels(config['class_names'])
        ax.set_yticklabels(config['class_names'])

    # 隐藏多余的子图
    for idx in range(n_models, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(config['output_dir'], 'confusion_matrices_comparison.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"混淆矩阵对比已保存到: {save_path}")


def plot_class_wise_performance(unified_results, config):
    """绘制各类别的性能对比"""

    # 准备数据
    models = list(unified_results.keys())
    classes = config['class_names']
    n_classes = len(classes)

    # 提取每个模型每个类别的敏感度和特异度
    sensitivity_data = []
    specificity_data = []

    for model_name, result in unified_results.items():
        metrics = result['metrics']
        sensitivity_data.append(metrics['sensitivity_per_class'])
        specificity_data.append(metrics['specificity_per_class'])

    sensitivity_data = np.array(sensitivity_data)  # [n_models, n_classes]
    specificity_data = np.array(specificity_data)

    # 绘制敏感度对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(n_classes)
    width = 0.15
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']

    # 敏感度
    for i, model_name in enumerate(models):
        ax1.bar(x + i * width, sensitivity_data[i], width,
                label=model_name, color=colors[i % len(colors)])

    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Sensitivity')
    ax1.set_title('Class-wise Sensitivity Comparison')
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(classes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # 特异度
    for i, model_name in enumerate(models):
        ax2.bar(x + i * width, specificity_data[i], width,
                label=model_name, color=colors[i % len(colors)])

    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Specificity')
    ax2.set_title('Class-wise Specificity Comparison')
    ax2.set_xticks(x + width * (len(models) - 1) / 2)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = os.path.join(config['output_dir'], 'class_wise_performance.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"类别性能对比已保存到: {save_path}")


def plot_model_performance_radar(unified_results, config):
    """绘制模型性能雷达图"""

    # 准备数据
    metrics = ['Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'AUC']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']

    for idx, (model_name, result) in enumerate(unified_results.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # 提取指标值
        model_metrics = result['metrics']
        values = [
            model_metrics['accuracy'],
            model_metrics['f1_macro'],
            model_metrics['sensitivity_mean'],
            model_metrics['specificity_mean'],
            model_metrics['auc']
        ]

        # 角度设置
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]

        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2,
                label=model_name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model_name} Performance', size=14, weight='bold')
        ax.grid(True)

        # 添加数值标签
        for angle, value, metric in zip(angles[:-1], values[:-1], metrics):
            ax.text(angle, value + 0.05, f'{value:.3f}',
                    ha='center', va='center', fontsize=10)

    # 隐藏多余的子图
    for idx in range(len(unified_results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(config['output_dir'], 'model_performance_radar.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"模型性能雷达图已保存到: {save_path}")


def calc_ap_bootstrap(y_true, y_pred, y_prob):
    """计算AP的Bootstrap函数"""
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize

    n_classes = len(np.unique(y_true))
    if n_classes > 2:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        class_aps = []
        for i in range(n_classes):
            ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            class_aps.append(ap)
        return np.mean(class_aps)
    else:
        return average_precision_score(y_true, y_prob[:, 1])


def generate_bootstrap_comparison_table(bootstrap_results, config):
    """生成Bootstrap结果对比表格"""

    # 添加空值检查
    if bootstrap_results is None:
        print("❌ Bootstrap结果为空，无法生成表格")
        return None

    print("\n" + "=" * 80)
    print("🔄 Bootstrap估计结果表（500次重采样均值 ± 95% CI）")
    print("=" * 80)

    bootstrap_data = []
    metrics_list = ['Accuracy', 'F1', 'Sensitivity', 'Specificity', 'AUC', 'AP']

    for model_name, results in bootstrap_results.items():
        row = {'Model': model_name}
        for metric in metrics_list:
            if metric in results:
                mean, lower, upper = results[metric]
                row[metric] = f"{mean:.4f} ({lower:.4f}-{upper:.4f})"
            else:
                row[metric] = "N/A"
        bootstrap_data.append(row)

    bootstrap_df = pd.DataFrame(bootstrap_data)
    print(bootstrap_df.to_string(index=False))

    # 保存表格
    bootstrap_path = os.path.join(config['output_dir'], 'bootstrap_metrics_comparison.csv')
    excel_path = os.path.join(config['output_dir'], 'bootstrap_metrics_comparison.xlsx')

    bootstrap_df.to_csv(bootstrap_path, index=False)
    bootstrap_df.to_excel(excel_path, index=False)

    print(f"\n📄 Bootstrap结果表格已保存到:")
    print(f"  - {bootstrap_path}")
    print(f"  - {excel_path}")

    return bootstrap_df


def generate_methodology_report(config):
    """生成方法学说明报告"""

    report_path = os.path.join(config['output_dir'], 'methodology_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Bootstrap方法学说明\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. Bootstrap重采样方法\n")
        f.write("-" * 30 + "\n")
        f.write("- 重采样次数: 500次\n")
        f.write("- 置信区间: 95%\n")
        f.write("- 采样方式: 有放回随机采样\n")
        f.write("- 样本大小: 与原始测试集相同\n\n")

        f.write("2. 指标计算说明\n")
        f.write("-" * 30 + "\n")
        f.write("- Bootstrap指标: 500次重采样的均值\n")
        f.write("- 置信区间: 2.5%和97.5%分位数\n")
        f.write("- 用途: 评估指标的稳定性和统计显著性\n\n")

        f.write("3. 结果解释\n")
        f.write("-" * 30 + "\n")
        f.write("- 柱状图显示: Bootstrap均值 ± 95%置信区间\n")
        f.write("- 表格显示: Bootstrap估计及置信区间\n")
        f.write("- 置信区间宽度: 反映指标的不确定性\n")
        f.write("- 模型比较: 基于置信区间重叠程度\n\n")

        f.write("4. 统计显著性判断\n")
        f.write("-" * 30 + "\n")
        f.write("- 置信区间不重叠: 模型间差异显著\n")
        f.write("- 置信区间重叠: 模型间差异可能不显著\n")
        f.write("- 置信区间宽度: 反映估计的不确定性\n")
        f.write("- 建议: 结合实际应用场景选择模型\n")

    print(f"📖 方法学说明已保存到: {report_path}")


def generate_final_summary_report_bootstrap(unified_results, bootstrap_results, config):
    """基于Bootstrap结果生成最终总结报告"""

    # 添加空值检查
    if bootstrap_results is None:
        print("❌ Bootstrap结果为空，使用原始结果生成报告")
        generate_final_summary_report(unified_results, config)
        return

    print("\n" + "=" * 80)
    print("🎯 DYNAMIC WEIGHT FUSION 最终总结报告 (Bootstrap估计)")
    print("=" * 80)

    # 创建性能对比表（基于Bootstrap结果）
    performance_table = PrettyTable()
    performance_table.field_names = [
        "Model", "Accuracy (95% CI)", "F1 (95% CI)", "Sensitivity (95% CI)", "Specificity (95% CI)", "AUC (95% CI)"
    ]

    best_metrics = {
        'accuracy': ('', 0),
        'f1': ('', 0),
        'sensitivity': ('', 0),
        'specificity': ('', 0),
        'auc': ('', 0)
    }

    for model_name, results in bootstrap_results.items():
        # 更新最佳指标（基于Bootstrap均值）
        for metric_name, bootstrap_key in [
            ('accuracy', 'Accuracy'),
            ('f1', 'F1'),
            ('sensitivity', 'Sensitivity'),
            ('specificity', 'Specificity'),
            ('auc', 'AUC')
        ]:
            if bootstrap_key in results:
                current_value = results[bootstrap_key][0]  # Bootstrap均值
                if current_value > best_metrics[metric_name][1]:
                    best_metrics[metric_name] = (model_name, current_value)

        # 添加到表格（显示Bootstrap结果）
        acc_mean, acc_lower, acc_upper = results['Accuracy']
        f1_mean, f1_lower, f1_upper = results['F1']
        sens_mean, sens_lower, sens_upper = results['Sensitivity']
        spec_mean, spec_lower, spec_upper = results['Specificity']
        auc_mean, auc_lower, auc_upper = results['AUC']

        performance_table.add_row([
            model_name,
            f"{acc_mean:.4f} ({acc_lower:.4f}-{acc_upper:.4f})",
            f"{f1_mean:.4f} ({f1_lower:.4f}-{f1_upper:.4f})",
            f"{sens_mean:.4f} ({sens_lower:.4f}-{sens_upper:.4f})",
            f"{spec_mean:.4f} ({spec_lower:.4f}-{spec_upper:.4f})",
            f"{auc_mean:.4f} ({auc_lower:.4f}-{auc_upper:.4f})"
        ])

    print("\n📈 所有模型性能对比 (Bootstrap估计):")
    print(performance_table)

    print("\n🏆 最佳性能指标 (Bootstrap均值):")
    metric_display = {
        'accuracy': '准确率',
        'f1': 'F1分数',
        'sensitivity': '敏感度',
        'specificity': '特异度',
        'auc': 'AUC'
    }
    for metric_name, (model_name, value) in best_metrics.items():
        print(f"  {metric_display[metric_name]}: {model_name} ({value:.4f})")

    # Dynamic Fusion的优势分析
    if 'DynamicFusion' in bootstrap_results:
        fusion_results = bootstrap_results['DynamicFusion']

        print(f"\n🚀 Dynamic Weight Fusion 性能分析 (Bootstrap估计):")
        print(f"  - 准确率: {fusion_results['Accuracy'][0]:.4f} ({fusion_results['Accuracy'][1]:.4f}-{fusion_results['Accuracy'][2]:.4f})")
        print(f"  - F1分数: {fusion_results['F1'][0]:.4f} ({fusion_results['F1'][1]:.4f}-{fusion_results['F1'][2]:.4f})")
        print(f"  - AUC: {fusion_results['AUC'][0]:.4f} ({fusion_results['AUC'][1]:.4f}-{fusion_results['AUC'][2]:.4f})")

        # 与最佳单模型对比
        best_single_f1 = 0
        best_single_model = ""
        for model_name, results in bootstrap_results.items():
            if model_name != 'DynamicFusion':
                if results['F1'][0] > best_single_f1:
                    best_single_f1 = results['F1'][0]
                    best_single_model = model_name

        improvement = fusion_results['F1'][0] - best_single_f1
        print(f"\n💡 相比最佳单模型 ({best_single_model}):")
        print(f"  - F1分数提升: {improvement:+.4f}")
        print(f"  - 相对提升: {(improvement / best_single_f1) * 100:+.2f}%")

    # 保存报告到文件
    report_path = os.path.join(config['output_dir'], 'final_summary_report_bootstrap.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Dynamic Weight Fusion 最终总结报告 (Bootstrap估计)\n")
        f.write("=" * 80 + "\n\n")

        f.write("模型架构:\n")
        f.write(f"- 图像模型: {', '.join(config['models'])}\n")
        f.write(f"- 黄疸特征分类器: 8维特征输入\n")
        f.write(f"- 融合方法: Dynamic Weight Fusion\n")
        f.write(f"- 类别数: {len(config['class_names'])}\n")
        f.write(f"- 评估方法: Bootstrap重采样 (500次迭代)\n\n")

        f.write("性能对比 (Bootstrap估计):\n")
        f.write(str(performance_table) + "\n\n")

        f.write("最佳性能指标 (Bootstrap均值):\n")
        for metric_name, (model_name, value) in best_metrics.items():
            f.write(f"  {metric_display[metric_name]}: {model_name} ({value:.4f})\n")

        if 'DynamicFusion' in bootstrap_results:
            fusion_results = bootstrap_results['DynamicFusion']
            f.write(f"\nDynamic Weight Fusion 详细性能 (Bootstrap估计):\n")
            f.write(f"  - 准确率: {fusion_results['Accuracy'][0]:.4f} ({fusion_results['Accuracy'][1]:.4f}-{fusion_results['Accuracy'][2]:.4f})\n")
            f.write(f"  - F1分数: {fusion_results['F1'][0]:.4f} ({fusion_results['F1'][1]:.4f}-{fusion_results['F1'][2]:.4f})\n")
            f.write(f"  - AUC: {fusion_results['AUC'][0]:.4f} ({fusion_results['AUC'][1]:.4f}-{fusion_results['AUC'][2]:.4f})\n")

            improvement = fusion_results['F1'][0] - best_single_f1
            f.write(f"\n相比最佳单模型提升:\n")
            f.write(f"  - F1分数提升: {improvement:+.4f}\n")
            f.write(f"  - 相对提升: {(improvement / best_single_f1) * 100:+.2f}%\n")

    print(f"\n📄 详细报告已保存到: {report_path}")

    print("\n🎉 Dynamic Weight Fusion 训练和评估完成!")
    print("📁 生成的文件包括:")
    print("  - unified_model_comparison_with_ap.svg: Bootstrap置信区间柱状图（含AP）")
    print("  - bootstrap_metrics_comparison.csv/xlsx: Bootstrap结果表格")
    print("  - comprehensive_roc_curves.svg: 综合ROC曲线")
    print("  - comprehensive_prc_curves.svg: 综合PRC曲线")
    print("  - [ModelName]_roc_curves.svg: 各模型单独ROC曲线")
    print("  - [ModelName]_prc_curves.svg: 各模型单独PRC曲线")
    print("  - confusion_matrices_comparison.svg: 混淆矩阵对比")
    print("  - class_wise_performance.svg: 类别性能对比")
    print("  - model_performance_radar.svg: 模型性能雷达图")
    print("  - methodology_report.txt: 方法学说明")
    print("  - final_summary_report_bootstrap.txt: 最终总结报告")


def train_and_evaluate_dynamic_fusion_comprehensive(config):
    """完整的训练和评估 - 包含所有可视化分析"""

    # 首先运行基础的Dynamic Weight Fusion训练
    print("🔥 开始Dynamic Weight Fusion基础训练...")
    fusion_model, test_results = train_and_evaluate_dynamic_fusion(config)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试数据加载器
    test_dir = os.path.join(config['dataset_dir'], "test")
    test_names = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

    full_test_dataset = Jaundice3ClassDataset_Full(
        test_dir, test_names, split='test',
        bilirubin_csv=config['bilirubin_csv'],
        transform=T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    full_test_loader = DataLoader(
        full_test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    # 加载所有模型
    print("\n📦 加载所有基础模型...")

    # 加载图像模型
    image_models = {}
    for model_name in config['models']:
        print(f"加载 {model_name}...")

        if model_name == 'ConvNext':
            model = ConvNextBackbone(model_name, num_classes=3, pretrained=True, freeze_base=True)
        elif model_name == 'Swin':
            model = SwinTransformerBackbone(model_name, num_classes=3, pretrained=True, freeze_base=True)
        elif model_name == 'EfficientNet':
            model = EfficientNetBackbone(model_name, num_classes=3, pretrained=True, freeze_base=True)
        elif model_name == 'ViT':
            model = ViTBackbone(model_name, num_classes=3, pretrained=True, freeze_base=True)

        model_path = config['existing_model_paths'][model_name]
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        image_models[model_name] = model
        print(f"✅ {model_name} 加载完成")

    # 加载黄疸模型
    print("加载黄疸特征分类器...")
    yellow_model = YellowFeatureClassifier(input_dim=8, hidden_dim=64, num_classes=3, dropout=0.3)
    yellow_model_path = os.path.join(config['output_dir'], 'best_yellow_classifier.pth')

    if not os.path.exists(yellow_model_path):
        print("❌ 未找到黄疸特征分类器，需要先训练...")
        raise FileNotFoundError(f"找不到黄疸分类器: {yellow_model_path}")

    yellow_model.load_state_dict(torch.load(yellow_model_path, map_location=device))
    yellow_model.to(device)
    yellow_model.eval()
    print("✅ 黄疸特征分类器加载完成")

    # 统一评估所有模型
    print("\n🔍 统一评估所有模型...")
    unified_results = unified_evaluate_all_models(
        image_models, yellow_model, fusion_model, full_test_loader, device, config
    )

    # 生成所有对比图表和分析
    print("\n📊 生成完整的可视化分析...")

    # 1. Bootstrap分析（包含AP）- 确保返回结果
    print("🔄 计算Bootstrap置信区间（500次重采样，包含AP）...")
    bootstrap_results = create_unified_bootstrap_chart_with_ap(unified_results, config)

    # 检查bootstrap_results是否成功返回
    if bootstrap_results is None:
        print("❌ Bootstrap分析失败，跳过相关步骤")
        return unified_results, None

    # 2. Bootstrap结果表格
    print("📋 生成Bootstrap结果表格...")
    bootstrap_df = generate_bootstrap_comparison_table(bootstrap_results, config)

    # 3. 综合ROC曲线
    print("📈 绘制综合ROC曲线...")
    plot_comprehensive_roc(unified_results, config)

    # 4. 综合PRC曲线
    print("📈 绘制综合PRC曲线...")
    macro_aps = plot_comprehensive_prc(unified_results, config)

    # 5. 单模型ROC曲线
    print("📈 绘制单模型ROC曲线...")
    plot_individual_model_roc(unified_results, config)

    # 6. 单模型PRC曲线
    print("📈 绘制单模型PRC曲线...")
    plot_individual_model_prc(unified_results, config)

    # 7. 混淆矩阵对比
    print("📊 绘制混淆矩阵对比...")
    plot_confusion_matrices(unified_results, config)

    # 8. 类别性能对比
    print("📊 绘制类别性能对比...")
    plot_class_wise_performance(unified_results, config)

    # 9. 模型性能雷达图
    print("📊 绘制模型性能雷达图...")
    plot_model_performance_radar(unified_results, config)

    # 10. 生成说明报告
    generate_methodology_report(config)

    # 11. 保存详细结果
    save_comprehensive_results(unified_results, config)

    # 12. 生成总结报告
    generate_final_summary_report_bootstrap(unified_results, bootstrap_results, config)

    return unified_results, bootstrap_results

def generate_final_summary_report(unified_results, config):
    """生成最终总结报告"""

    print("\n" + "=" * 80)
    print("🎯 DYNAMIC WEIGHT FUSION 最终总结报告")
    print("=" * 80)

    # 创建性能对比表
    performance_table = PrettyTable()
    performance_table.field_names = [
        "Model", "Accuracy", "F1 (Macro)", "Sensitivity", "Specificity", "AUC"
    ]

    best_metrics = {
        'accuracy': ('', 0),
        'f1_macro': ('', 0),
        'sensitivity_mean': ('', 0),
        'specificity_mean': ('', 0),
        'auc': ('', 0)
    }

    for model_name, result in unified_results.items():
        metrics = result['metrics']

        # 更新最佳指标
        for metric_name, current_value in [
            ('accuracy', metrics['accuracy']),
            ('f1_macro', metrics['f1_macro']),
            ('sensitivity_mean', metrics['sensitivity_mean']),
            ('specificity_mean', metrics['specificity_mean']),
            ('auc', metrics['auc'])
        ]:
            if current_value > best_metrics[metric_name][1]:
                best_metrics[metric_name] = (model_name, current_value)

        # 添加到表格
        performance_table.add_row([
            model_name,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['f1_macro']:.4f}",
            f"{metrics['sensitivity_mean']:.4f}",
            f"{metrics['specificity_mean']:.4f}",
            f"{metrics['auc']:.4f}"
        ])

    print("\n📈 所有模型性能对比:")
    print(performance_table)

    print("\n🏆 最佳性能指标:")
    for metric_name, (model_name, value) in best_metrics.items():
        metric_display = {
            'accuracy': '准确率',
            'f1_macro': 'F1分数',
            'sensitivity_mean': '敏感度',
            'specificity_mean': '特异度',
            'auc': 'AUC'
        }
        print(f"  {metric_display[metric_name]}: {model_name} ({value:.4f})")

    # Dynamic Fusion的优势分析
    if 'DynamicFusion' in unified_results:
        fusion_metrics = unified_results['DynamicFusion']['metrics']

        print(f"\n🚀 Dynamic Weight Fusion 性能分析:")
        print(f"  - 准确率: {fusion_metrics['accuracy']:.4f}")
        print(f"  - F1分数: {fusion_metrics['f1_macro']:.4f}")
        print(f"  - AUC: {fusion_metrics['auc']:.4f}")

        # 与最佳单模型对比
        best_single_f1 = 0
        best_single_model = ""
        for model_name, result in unified_results.items():
            if model_name != 'DynamicFusion':
                if result['metrics']['f1_macro'] > best_single_f1:
                    best_single_f1 = result['metrics']['f1_macro']
                    best_single_model = model_name

        improvement = fusion_metrics['f1_macro'] - best_single_f1
        print(f"\n💡 相比最佳单模型 ({best_single_model}):")
        print(f"  - F1分数提升: {improvement:+.4f}")
        print(f"  - 相对提升: {(improvement / best_single_f1) * 100:+.2f}%")

    # 保存报告到文件
    report_path = os.path.join(config['output_dir'], 'final_summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Dynamic Weight Fusion 最终总结报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("模型架构:\n")
        f.write(f"- 图像模型: {', '.join(config['models'])}\n")
        f.write(f"- 黄疸特征分类器: 8维特征输入\n")
        f.write(f"- 融合方法: Dynamic Weight Fusion\n")
        f.write(f"- 类别数: {len(config['class_names'])}\n\n")

        f.write("性能对比:\n")
        f.write(str(performance_table) + "\n\n")

        f.write("最佳性能指标:\n")
        for metric_name, (model_name, value) in best_metrics.items():
            metric_display = {
                'accuracy': '准确率',
                'f1_macro': 'F1分数',
                'sensitivity_mean': '敏感度',
                'specificity_mean': '特异度',
                'auc': 'AUC'
            }
            f.write(f"  {metric_display[metric_name]}: {model_name} ({value:.4f})\n")

        if 'DynamicFusion' in unified_results:
            fusion_metrics = unified_results['DynamicFusion']['metrics']
            f.write(f"\nDynamic Weight Fusion 详细性能:\n")
            f.write(f"  - 准确率: {fusion_metrics['accuracy']:.4f}\n")
            f.write(f"  - F1分数: {fusion_metrics['f1_macro']:.4f}\n")
            f.write(f"  - AUC: {fusion_metrics['auc']:.4f}\n")

            improvement = fusion_metrics['f1_macro'] - best_single_f1
            f.write(f"\n相比最佳单模型提升:\n")
            f.write(f"  - F1分数提升: {improvement:+.4f}\n")
            f.write(f"  - 相对提升: {(improvement / best_single_f1) * 100:+.2f}%\n")

    print(f"\n📄 详细报告已保存到: {report_path}")

    print("\n🎉 Dynamic Weight Fusion 训练和评估完成!")
    print("📁 生成的文件包括:")
    print("  - unified_model_comparison.svg: Bootstrap置信区间柱状图")
    print("  - comprehensive_roc_curves.svg: 综合ROC曲线")
    print("  - model_comparison_detailed.csv/xlsx: 详细对比表格")
    print("  - comprehensive_results.json: 完整结果数据")
    print("  - final_summary_report.txt: 最终总结报告")


def generate_comparison_table(unified_results, config):
    """生成模型对比表格"""

    # 创建DataFrame
    data = []
    for model_name, result in unified_results.items():
        metrics = result['metrics']
        data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1 (Macro)': f"{metrics['f1_macro']:.4f}",
            'F1 (Weighted)': f"{metrics['f1_weighted']:.4f}",
            'Sensitivity': f"{metrics['sensitivity_mean']:.4f}",
            'Specificity': f"{metrics['specificity_mean']:.4f}",
            'AUC': f"{metrics['auc']:.4f}"
        })

    df = pd.DataFrame(data)

    # 保存CSV和Excel
    csv_path = os.path.join(config['output_dir'], 'model_comparison_detailed.csv')
    excel_path = os.path.join(config['output_dir'], 'model_comparison_detailed.xlsx')

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    print(f"详细对比表格已保存到:")
    print(f"  - {csv_path}")
    print(f"  - {excel_path}")

    # 打印表格
    print("\n" + "=" * 80)
    print("模型性能对比表")
    print("=" * 80)
    print(df.to_string(index=False))


def train_and_evaluate_dynamic_fusion(config):
    """主训练函数：专注于logits融合的集成学习"""

    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(config['output_dir'], exist_ok=True)

    # 数据准备
    train_dir = os.path.join(config['dataset_dir'], "train")
    val_dir = os.path.join(config['dataset_dir'], "val")
    test_dir = os.path.join(config['dataset_dir'], "test")

    train_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    val_names = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    test_names = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

    print(f"数据集大小 - Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")

    # =================== 加载已训练的基础模型 ===================
    print("\n" + "=" * 50)
    print("加载已训练的基础模型")
    print("=" * 50)

    # 加载图像模型
    image_models = {}
    transforms = {
        'convnext': T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
                'swin': T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'efficientnet': T.Compose([
            T.Resize((384, 384)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'vit': T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    for model_name in config['models']:
        print(f"加载 {model_name} 模型...")

        # 创建模型实例
        if model_name == 'ConvNext':
            model = ConvNextBackbone(model_name, num_classes=3, pretrained=True, freeze_base=True)
        elif model_name == 'Swin':
            model = SwinTransformerBackbone(model_name, num_classes=3, pretrained=True, freeze_base=True)
        elif model_name == 'EfficientNet':
            model = EfficientNetBackbone(model_name, num_classes=3, pretrained=True, freeze_base=True)
        elif model_name == 'ViT':
            model = ViTBackbone(model_name, num_classes=3, pretrained=True, freeze_base=True)

        # 加载预训练权重
        model_path = config['existing_model_paths'][model_name]
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            image_models[model_name] = model
            print(f"✓ {model_name} 模型加载成功")
        else:
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

    # 加载黄疸特征模型
    print("加载黄疸特征分类器...")
    yellow_model = YellowFeatureClassifier(input_dim=8, hidden_dim=64, num_classes=3, dropout=0.3)
    yellow_model_path = os.path.join(config['output_dir'], 'best_yellow_classifier.pth')

    if os.path.exists(yellow_model_path):
        yellow_model.load_state_dict(torch.load(yellow_model_path, map_location=device))
        yellow_model.to(device)
        yellow_model.eval()
        print("✓ 黄疸特征分类器加载成功")
    else:
        print("未找到黄疸特征分类器，需要先训练...")
        # 这里可以调用训练黄疸分类器的函数
        raise FileNotFoundError(f"找不到黄疸分类器: {yellow_model_path}")

    # =================== 创建数据加载器 ===================
    print("\n创建数据加载器...")

    # 训练集
    train_dataset = Jaundice3ClassDataset_Full(
        train_dir, train_names, split='train',
        bilirubin_csv=config['bilirubin_csv'],
        transform=transforms['convnext'],  # 使用统一变换
        enable_undersample=True,
        undersample_count=200
    )

    # 验证集
    val_dataset = Jaundice3ClassDataset_Full(
        val_dir, val_names, split='val',
        bilirubin_csv=config['bilirubin_csv'],
        transform=transforms['convnext']
    )

    # 测试集
    test_dataset = Jaundice3ClassDataset_Full(
        test_dir, test_names, split='test',
        bilirubin_csv=config['bilirubin_csv'],
        transform=transforms['convnext']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'])

    print(f"✓ 数据加载器创建完成")
    print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}, 测试批次数: {len(test_loader)}")

    # =================== 训练Logits融合集成模型 ===================
    print("\n" + "=" * 50)
    print("提取所有模型的logits并训练Dynamic Weight Fusion")
    print("=" * 50)

    # 提取训练数据logits
    print("提取训练数据logits...")
    train_logits, train_labels = extract_all_logits(train_loader, image_models, yellow_model, device)

    print("提取验证数据logits...")
    val_logits, val_labels = extract_all_logits(val_loader, image_models, yellow_model, device)

    # 训练Dynamic Weight Fusion
    print("训练Dynamic Weight Fusion模型...")
    fusion_model = train_dynamic_fusion_simple(
        train_logits, train_labels, val_logits, val_labels, device, config
    )

    # 测试评估
    print("提取测试数据logits...")
    test_logits, test_labels = extract_all_logits(test_loader, image_models, yellow_model, device)

    print("评估Dynamic Weight Fusion模型...")
    test_results = evaluate_dynamic_fusion(fusion_model, test_logits, test_labels, device, config)

    return fusion_model, test_results


def calc_ap_bootstrap(y_true, y_pred, y_prob):
    """计算AP的Bootstrap函数"""
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize

    n_classes = len(np.unique(y_true))
    if n_classes > 2:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        class_aps = []
        for i in range(n_classes):
            ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            class_aps.append(ap)
        return np.mean(class_aps)
    else:
        return average_precision_score(y_true, y_prob[:, 1])


def create_unified_bootstrap_chart_with_ap(unified_results, config):
    """创建包含AP的统一Bootstrap柱状图"""

    print("🔄 开始Bootstrap重采样分析（包含AP指标）...")

    metrics = ['Accuracy', 'F1', 'Sensitivity', 'Specificity', 'AUC', 'AP']
    models = list(unified_results.keys())
    n_metrics = len(metrics)
    n_models = len(models)

    # 计算Bootstrap置信区间
    bootstrap_results = {}

    try:
        for model_name, result in unified_results.items():
            print(f"  处理模型: {model_name}")
            y_true, y_pred, y_prob = result['predictions']
            bootstrap_results[model_name] = {}

            for metric in metrics:
                try:
                    if metric == 'Accuracy':
                        mean, (lower, upper) = bootstrap_metric(
                            y_true, y_pred, None,
                            lambda y_true, y_pred, *args: accuracy_score(y_true, y_pred)
                        )
                    elif metric == 'F1':
                        mean, (lower, upper) = bootstrap_metric(
                            y_true, y_pred, None,
                            lambda y_true, y_pred, *args: f1_score(y_true, y_pred, average='macro')
                        )
                    elif metric == 'Sensitivity':
                        mean, (lower, upper) = bootstrap_metric(
                            y_true, y_pred, None, calc_sensitivity_bootstrap
                        )
                    elif metric == 'Specificity':
                        mean, (lower, upper) = bootstrap_metric(
                            y_true, y_pred, None, calc_specificity_bootstrap
                        )
                    elif metric == 'AUC':
                        mean, (lower, upper) = bootstrap_metric(
                            y_true, y_pred, y_prob, calc_auc_bootstrap
                        )
                    elif metric == 'AP':
                        mean, (lower, upper) = bootstrap_metric(
                            y_true, y_pred, y_prob, calc_ap_bootstrap
                        )

                    bootstrap_results[model_name][metric] = (mean, lower, upper)
                    print(f"    ✓ {metric}: {mean:.4f} ({lower:.4f}-{upper:.4f})")

                except Exception as e:
                    print(f"    ❌ {metric} 计算失败: {e}")
                    bootstrap_results[model_name][metric] = (0.0, 0.0, 0.0)

        # 绘制柱状图
        plt.figure(figsize=(14, 8))
        bar_width = 0.12
        index = np.arange(n_models)
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#FF7F0E']

        for i, metric in enumerate(metrics):
            means = []
            lowers = []
            uppers = []

            for model_name in models:
                mean, lower, upper = bootstrap_results[model_name][metric]
                means.append(mean)
                lowers.append(lower)
                uppers.append(upper)

            means = np.array(means)
            lowers = np.array(lowers)
            uppers = np.array(uppers)
            yerr = np.vstack([means - lowers, uppers - means])

            plt.bar(
                index + i * bar_width,
                means,
                bar_width,
                label=metric,
                color=colors[i % len(colors)],
                yerr=yerr,
                capsize=3,
                edgecolor='black',
                linewidth=0.8,
                error_kw={'elinewidth': 1.0, 'capthick': 1.0}
            )

            # 添加数值标签
            for j, value in enumerate(means):
                plt.text(
                    j + i * bar_width,
                    uppers[j] + 0.01,
                    f'{value:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=90
                )

        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Bootstrap Mean ± 95% CI', fontsize=12)
        plt.title('Comprehensive Model Comparison with AP (Bootstrap 95% CI)', fontsize=14)
        plt.xticks(index + bar_width * (n_metrics - 1) / 2, models, rotation=15)
        plt.ylim(0, 1.05)
        plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=6)
        plt.tight_layout(rect=[0, 0.07, 1, 1])

        save_path = os.path.join(config['output_dir'], 'unified_model_comparison_with_ap.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()

        print(f"✅ 包含AP的统一柱状图已保存到: {save_path}")
        print(f"✅ Bootstrap分析完成，返回 {len(bootstrap_results)} 个模型的结果")

        return bootstrap_results

    except Exception as e:
        print(f"❌ Bootstrap分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None



def plot_multiclass_prc(y_true, y_score, class_names, save_path=None):
    """绘制多类别PRC曲线"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import seaborn as sns

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(10, 8))
    colors = custom_palette[:n_classes]

    # 计算每个类别的PRC
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])

        plt.plot(recall[i], precision[i],
                 label=f'{class_names[i]} (AP={average_precision[i]:.3f})',
                 color=colors[i], linewidth=2)

    # 计算macro-average PRC
    all_y_true = y_true_bin.ravel()
    all_y_scores = y_score.ravel()
    macro_precision, macro_recall, _ = precision_recall_curve(all_y_true, all_y_scores)
    macro_ap = average_precision_score(all_y_true, all_y_scores)

    plt.plot(macro_recall, macro_precision,
             label=f'Macro-average (AP={macro_ap:.3f})',
             color='black', linestyle='--', linewidth=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall Curve')
    plt.legend(loc="lower left", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        if save_path.endswith('.png'):
            save_path = save_path.replace('.png', '.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"PRC曲线已保存到: {save_path}")

    plt.close()

    return average_precision


def plot_comprehensive_prc(unified_results, config):
    """绘制所有模型的综合PRC曲线对比"""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    plt.figure(figsize=(12, 8))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']

    macro_aps = {}  # 存储每个模型的macro-average AP

    for i, (model_name, result) in enumerate(unified_results.items()):
        y_true, y_pred, y_prob = result['predictions']

        # 计算macro-average PRC
        n_classes = len(config['class_names'])
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        # 方法1：计算每个类别的AP然后平均
        class_aps = []
        for class_idx in range(n_classes):
            ap = average_precision_score(y_true_bin[:, class_idx], y_prob[:, class_idx])
            class_aps.append(ap)

        macro_ap = np.mean(class_aps)
        macro_aps[model_name] = macro_ap

        # 计算macro-average precision-recall曲线
        all_y_true = y_true_bin.ravel()
        all_y_scores = y_prob.ravel()
        macro_precision, macro_recall, _ = precision_recall_curve(all_y_true, all_y_scores)

        plt.plot(macro_recall, macro_precision,
                 label=f'{model_name} (mAP = {macro_ap:.3f})',
                 color=colors[i % len(colors)], linewidth=2)

    # 添加基线（随机分类器）
    baseline_ap = np.mean([np.sum(y_true == i) / len(y_true) for i in range(n_classes)])
    plt.axhline(y=baseline_ap, color='gray', linestyle=':',
                label=f'Random Baseline (AP = {baseline_ap:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Comprehensive Precision-Recall Curves - All Models')
    plt.legend(loc="lower left", frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(config['output_dir'], 'comprehensive_prc_curves.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"综合PRC曲线已保存到: {save_path}")
    return macro_aps


def plot_class_specific_prc(unified_results, config):
    """为每个类别绘制详细的PRC曲线"""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    n_classes = len(config['class_names'])
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']

    class_aps = {class_name: {} for class_name in config['class_names']}

    for class_idx, class_name in enumerate(config['class_names']):
        ax = axes[class_idx]

        for model_idx, (model_name, result) in enumerate(unified_results.items()):
            y_true, y_pred, y_prob = result['predictions']

            # 二值化标签
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

            # 计算该类别的PRC
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, class_idx], y_prob[:, class_idx]
            )
            ap = average_precision_score(y_true_bin[:, class_idx], y_prob[:, class_idx])

            class_aps[class_name][model_name] = ap

            ax.plot(recall, precision,
                    label=f'{model_name} (AP={ap:.3f})',
                    color=colors[model_idx % len(colors)], linewidth=2)

        # 添加基线
        baseline_ap = np.sum(y_true == class_idx) / len(y_true)
        ax.axhline(y=baseline_ap, color='gray', linestyle=':',
                   label=f'Baseline (AP={baseline_ap:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{class_name} - Precision-Recall Curve')
        ax.legend(loc="lower left", frameon=False, fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config['output_dir'], 'class_specific_prc_curves.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"类别专用PRC曲线已保存到: {save_path}")
    return class_aps


def calc_ap_bootstrap(y_true, y_pred, y_prob):
    """计算AP的Bootstrap函数 - 安全版"""
    try:
        from sklearn.metrics import average_precision_score
        from sklearn.preprocessing import label_binarize

        # 确保输入是numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # 获取类别数
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)

        if n_classes > 2:
            # 多分类情况
            y_true_bin = label_binarize(y_true, classes=unique_classes)

            # 确保y_prob的维度正确
            if y_prob.ndim == 1:
                raise ValueError("y_prob应该是2D数组，对于多分类问题")

            class_aps = []
            for i in range(n_classes):
                try:
                    ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                    class_aps.append(ap)
                except Exception as e:
                    print(f"    警告: 类别 {i} 的AP计算失败: {e}")
                    class_aps.append(0.0)

            return np.mean(class_aps)
        else:
            # 二分类情况
            if y_prob.ndim == 2:
                return average_precision_score(y_true, y_prob[:, 1])
            else:
                return average_precision_score(y_true, y_prob)

    except Exception as e:
        print(f"    AP计算失败: {e}")
        return 0.0


# Quick test function to debug the issue
def test_bootstrap_inputs(unified_results):
    """测试Bootstrap输入数据的有效性"""
    print("\n🔍 检查Bootstrap输入数据...")

    for model_name, result in unified_results.items():
        print(f"\n模型: {model_name}")

        try:
            y_true, y_pred, y_prob = result['predictions']

            print(f"  y_true shape: {np.array(y_true).shape}")
            print(f"  y_pred shape: {np.array(y_pred).shape}")
            print(f"  y_prob shape: {np.array(y_prob).shape}")
            print(f"  Unique classes in y_true: {np.unique(y_true)}")
            print(f"  Unique classes in y_pred: {np.unique(y_pred)}")

            # 检查y_prob的有效性
            y_prob_array = np.array(y_prob)
            if y_prob_array.ndim == 2:
                print(f"  y_prob sum per row (should be ~1.0): {y_prob_array.sum(axis=1)[:5]}...")

        except Exception as e:
            print(f"  ❌ 数据检查失败: {e}")


def evaluate_logits_fusion_ensemble(image_models, yellow_model, ensemble_model,
                                   test_loader, device, config):
    """评估logits融合集成模型"""

    print("开始测试集评估...")

    # 设置所有模型为评估模式
    for model in image_models.values():
        model.eval()
    yellow_model.eval()
    ensemble_model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_attention_weights = []

    # 用于分析各个模型的贡献
    individual_logits = {name: [] for name in list(image_models.keys()) + ['YellowFeatures']}

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="测试集评估")

        for batch in test_pbar:
            images = batch['image'].to(device)
            yellow_features = batch['yellow_features'].to(device)
            labels = batch['label'].to(device)

            # 获取所有模型的logits
            logits_list = []
            model_names = list(image_models.keys()) + ['YellowFeatures']

            # 图像模型logits
            for i, (model_name, model) in enumerate(image_models.items()):
                _, logits = model(images)
                logits_list.append(logits)
                individual_logits[model_name].append(logits.cpu())

            # 黄疸特征模型logits
            yellow_logits = yellow_model(yellow_features)
            logits_list.append(yellow_logits)
            individual_logits['YellowFeatures'].append(yellow_logits.cpu())

            # 集成融合
            if config['ensemble']['fusion_method'] == 'mlp_with_yellow':
                fused_logits, attention_weights = ensemble_model(logits_list, yellow_features)
            else:
                fused_logits, attention_weights = ensemble_model(logits_list)

            # 获取预测结果
            probabilities = torch.softmax(fused_logits, dim=1)
            _, predicted = torch.max(fused_logits, 1)

            # 收集结果
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算详细指标
    results = calculate_unified_metrics(all_labels, all_preds, all_probs, config['class_names'])

    # 打印结果
    print(f"\n{'='*20} 测试集最终结果 {'='*20}")
    print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"F1分数 (Macro): {results['f1_macro']:.4f}")
    print(f"F1分数 (Weighted): {results['f1_weighted']:.4f}")
    print(f"精确率 (Macro): {results['precision_macro']:.4f}")
    print(f"召回率 (Macro): {results['recall_macro']:.4f}")
    print(f"敏感度 (平均): {results['sensitivity_mean']:.4f}")
    print(f"特异度 (平均): {results['specificity_mean']:.4f}")
    print(f"AUC: {results['auc']:.4f}")

    # 每类别详细结果
    print(f"\n每类别详细结果:")
    table = PrettyTable()
    table.field_names = ["类别", "敏感度", "特异度"]
    for i, class_name in enumerate(config['class_names']):
        table.add_row([
            class_name,
            f"{results['sensitivity_per_class'][i]:.4f}",
            f"{results['specificity_per_class'][i]:.4f}"
        ])
    print(table)

    # 混淆矩阵
    print(f"\n混淆矩阵:")
    cm_table = PrettyTable()
    cm_table.field_names = ["实际\\预测"] + [f"Pred_{name}" for name in config['class_names']]
    for i, row in enumerate(results['confusion_matrix']):
        cm_table.add_row([f"True_{config['class_names'][i]}"] + list(row))
    print(cm_table)

    # 分析各模型贡献度
    analyze_model_contributions(individual_logits, all_labels, config)

    # 绘制ROC曲线
    roc_save_path = os.path.join(config['output_dir'], 'ensemble_roc_curve.svg')
    plot_multiclass_roc(all_labels, all_probs, config['class_names'], save_path=roc_save_path)

    return {
        'metrics': results,
        'predictions': (all_labels, all_preds, all_probs),
        'individual_logits': individual_logits
    }



def analyze_model_contributions(individual_logits, labels, config):
    """分析各个模型的贡献度"""

    print(f"\n{'='*20} 模型贡献度分析 {'='*20}")

    model_performances = {}

    for model_name, logits_list in individual_logits.items():
        # 合并所有logits
        all_logits = torch.cat(logits_list, dim=0)

        # 计算单独预测
        _, predictions = torch.max(all_logits, 1)
        predictions = predictions.numpy()

        # 计算性能指标
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')

        model_performances[model_name] = {
            'accuracy': accuracy,
            'f1': f1
        }

    # 打印结果
    print("各模型单独性能:")
    perf_table = PrettyTable()
    perf_table.field_names = ["模型", "准确率", "F1分数"]

    for model_name, perf in model_performances.items():
        perf_table.add_row([
            model_name,
            f"{perf['accuracy']:.4f}",
            f"{perf['f1']:.4f}"
        ])

    print(perf_table)

    # 绘制性能对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    models = list(model_performances.keys())
    accuracies = [model_performances[m]['accuracy'] for m in models]
    f1_scores = [model_performances[m]['f1'] for m in models]

    # 准确率对比
    bars1 = ax1.bar(models, accuracies, color=custom_palette[:len(models)])
    ax1.set_title('Individual Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # 添加数值标签
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')

    # F1分数对比
    bars2 = ax2.bar(models, f1_scores, color=custom_palette[:len(models)])
    ax2.set_title('Individual Model F1 Score')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # 添加数值标签
    for bar, f1 in zip(bars2, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    save_path = os.path.join(config['output_dir'], 'individual_model_performance.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"模型性能对比图已保存到: {save_path}")


def generate_logits_fusion_report(image_models, yellow_model, ensemble_model,
                                 test_results, training_history, config):
    """生成logits融合集成的详细报告"""

    report_path = os.path.join(config['output_dir'], 'logits_fusion_ensemble_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Logits融合集成学习详细报告\n")
        f.write("=" * 60 + "\n\n")

        # 1. 架构信息
        f.write("1. 模型架构\n")
        f.write("-" * 30 + "\n")
        f.write(f"基础图像模型数量: {len(image_models)}\n")
        f.write(f"图像模型列表: {list(image_models.keys())}\n")
        f.write(f"黄疸特征模型: YellowFeatureClassifier (8维输入)\n")
        f.write(f"融合方法: {config['ensemble']['fusion_method']}\n")

        if config['ensemble']['fusion_method'] == 'self_attention':
            f.write(f"Self Attention参数:\n")
            f.write(f"  - d_model: {config['ensemble']['d_model']}\n")
            f.write(f"  - num_heads: {config['ensemble']['num_heads']}\n")
            f.write(f"  - dropout: {config['ensemble']['dropout']}\n")
            f.write(f"  - pooling_strategy: {config['ensemble']['pooling_strategy']}\n")

        f.write("\n")

        # 2. 训练配置
        f.write("2. 训练配置\n")
        f.write("-" * 30 + "\n")
        f.write(f"批次大小: {config['batch_size']}\n")
        f.write(f"学习率: {config['ensemble']['lr']}\n")
        f.write(f"权重衰减: {config['ensemble']['weight_decay']}\n")
        f.write(f"最大训练轮数: {config['ensemble']['epochs']}\n")
        f.write(f"早停耐心值: {config['ensemble']['patience']}\n")
        f.write(f"实际训练轮数: {len(training_history['train_loss'])}\n")
        f.write("\n")

        # 3. 最终性能
        f.write("3. 测试集最终性能\n")
        f.write("-" * 30 + "\n")
        metrics = test_results['metrics']
        f.write(f"准确率: {metrics['accuracy']:.4f}\n")
        f.write(f"F1分数 (Macro): {metrics['f1_macro']:.4f}\n")
        f.write(f"F1分数 (Weighted): {metrics['f1_weighted']:.4f}\n")
        f.write(f"精确率 (Macro): {metrics['precision_macro']:.4f}\n")
        f.write(f"召回率 (Macro): {metrics['recall_macro']:.4f}\n")
        f.write(f"敏感度 (平均): {metrics['sensitivity_mean']:.4f}\n")
        f.write(f"特异度 (平均): {metrics['specificity_mean']:.4f}\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write("\n")

        # 4. 每类别性能
        f.write("4. 每类别详细性能\n")
        f.write("-" * 30 + "\n")
        for i, class_name in enumerate(config['class_names']):
            f.write(f"{class_name}:\n")
            f.write(f"  敏感度: {metrics['sensitivity_per_class'][i]:.4f}\n")
            f.write(f"  特异度: {metrics['specificity_per_class'][i]:.4f}\n")
        f.write("\n")

        # 5. 训练历史统计
        f.write("5. 训练历史统计\n")
        f.write("-" * 30 + "\n")
        f.write(f"最佳训练F1: {max(training_history['train_f1']):.4f}\n")
        f.write(f"最佳验证F1: {max(training_history['val_f1']):.4f}\n")
        f.write(f"最终训练损失: {training_history['train_loss'][-1]:.4f}\n")
        f.write(f"最终验证损失: {training_history['val_loss'][-1]:.4f}\n")

        # 检查过拟合
        final_gap = training_history['train_f1'][-1] - training_history['val_f1'][-1]
        f.write(f"最终训练-验证F1差距: {final_gap:.4f}\n")
        if final_gap > 0.1:
            f.write("⚠️  检测到轻微过拟合\n")
        f.write("\n")

        # 6. 模型权重信息（如果适用）
        if config['ensemble']['fusion_method'] == 'weighted_average':
            f.write("6. 模型权重分配\n")
            f.write("-" * 30 + "\n")
            weights = torch.softmax(ensemble_model.model_weights, dim=0).cpu().numpy()
            model_names = list(image_models.keys()) + ['YellowFeatures']
            for name, weight in zip(model_names, weights):
                f.write(f"{name}: {weight:.4f}\n")
            f.write("\n")

        # 7. 技术特点
        f.write("7. 技术特点\n")
        f.write("-" * 30 + "\n")
        f.write("✓ 直接对logits进行融合，保留更多信息\n")
        f.write("✓ 使用早停机制防止过拟合\n")
        f.write("✓ 真正的Self Attention机制学习模型间关系\n")
        f.write("✓ 多种融合策略可选\n")
        f.write("✓ 详细的注意力权重分析\n")
        f.write("✓ 个体模型贡献度分析\n")

    print(f"\n📊 详细报告已保存到: {report_path}")

def calc_sensitivity_bootstrap(y_true, y_pred, y_prob):
    """计算敏感度的Bootstrap函数"""
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = np.diag(cm) / np.sum(cm, axis=1)
    return np.mean(sensitivities)


def calc_specificity_bootstrap(y_true, y_pred, y_prob):
    """计算特异度的Bootstrap函数"""
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    specificities = []
    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificities.append(spec)
    return np.mean(specificities)


def calc_auc_bootstrap(y_true, y_pred, y_prob):
    """计算AUC的Bootstrap函数"""
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    return roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')


def calculate_unified_metrics(y_true, y_pred, y_prob, class_names):
    """计算统一的评估指标（包含AP）"""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, roc_auc_score,
                                 average_precision_score)
    from sklearn.preprocessing import label_binarize

    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')

    # 混淆矩阵和敏感度/特异度
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)

    sensitivity = []
    specificity = []
    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0

        sensitivity.append(sens)
        specificity.append(spec)

    # AUC计算
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes > 2:
            auc_score = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
        else:
            auc_score = roc_auc_score(y_true, y_prob[:, 1])
    except:
        auc_score = 0.0

    # AP计算（新增）
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        if n_classes > 2:
            # 计算每个类别的AP
            class_aps = []
            for i in range(n_classes):
                ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                class_aps.append(ap)
            ap_macro = np.mean(class_aps)
            ap_weighted = average_precision_score(y_true_bin, y_prob, average='weighted')
        else:
            ap_macro = average_precision_score(y_true, y_prob[:, 1])
            ap_weighted = ap_macro
            class_aps = [ap_macro]
    except:
        ap_macro = 0.0
        ap_weighted = 0.0
        class_aps = [0.0] * n_classes

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'sensitivity_mean': np.mean(sensitivity),
        'specificity_mean': np.mean(specificity),
        'auc': auc_score,
        'ap_macro': ap_macro,  # 新增
        'ap_weighted': ap_weighted,  # 新增
        'ap_per_class': class_aps,  # 新增
        'sensitivity_per_class': sensitivity,
        'specificity_per_class': specificity,
        'confusion_matrix': cm
    }


def unified_evaluate_all_models(models, testloaders, ensemble_results, device, class_names):
    """统一评估所有模型，确保数据一致性"""

    all_results = {}

    # 1. 评估单个模型
    for model_name, model in models.items():
        print(f"Unified evaluation for {model_name}...")
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in testloaders[model_name]:
                if len(batch) == 4:  # 包含yellow features
                    inputs, _, labels, _ = batch
                else:  # 不包含yellow features
                    inputs, labels, _ = batch

                inputs = inputs.to(device)
                labels = labels.to(device)

                _, outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 转换为numpy数组
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        # 计算统一的指标
        metrics = calculate_unified_metrics(y_true, y_pred, y_prob, class_names)
        all_results[model_name] = {
            'metrics': metrics,
            'predictions': (y_true, y_pred, y_prob)
        }

    # 2. 添加Ensemble结果
    if ensemble_results is not None:
        y_test, test_pred, test_prob = ensemble_results
        metrics = calculate_unified_metrics(y_test, test_pred, test_prob, class_names)
        all_results['Ensemble'] = {
            'metrics': metrics,
            'predictions': (y_test, test_pred, test_prob)
        }

    return all_results


def create_unified_bootstrap_chart_corrected(unified_results, config):
    """修正版：明确使用Bootstrap结果，标题和标签都说明这一点"""

    metrics = ['Accuracy', 'F1', 'Sensitivity', 'Specificity', 'AUC']
    models = list(unified_results.keys())

    # 计算Bootstrap置信区间
    bootstrap_results = {}

    for model_name, result in unified_results.items():
        y_true, y_pred, y_prob = result['predictions']
        bootstrap_results[model_name] = {}

        for metric in metrics:
            if metric == 'Accuracy':
                mean, (lower, upper) = bootstrap_metric(
                    y_true, y_pred, None,
                    lambda y_true, y_pred, *args: accuracy_score(y_true, y_pred)
                )
            elif metric == 'F1':
                mean, (lower, upper) = bootstrap_metric(
                    y_true, y_pred, None,
                    lambda y_true, y_pred, *args: f1_score(y_true, y_pred, average='macro')
                )
            elif metric == 'Sensitivity':
                mean, (lower, upper) = bootstrap_metric(
                    y_true, y_pred, None, calc_sensitivity_bootstrap
                )
            elif metric == 'Specificity':
                mean, (lower, upper) = bootstrap_metric(
                    y_true, y_pred, None, calc_specificity_bootstrap
                )
            elif metric == 'AUC':
                mean, (lower, upper) = bootstrap_metric(
                    y_true, y_pred, y_prob, calc_auc_bootstrap
                )

            bootstrap_results[model_name][metric] = (mean, lower, upper)

    # 绘制柱状图
    plt.figure(figsize=(12, 8))
    bar_width = 0.15
    index = np.arange(len(models))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']

    for i, metric in enumerate(metrics):
        means = []
        lowers = []
        uppers = []

        for model_name in models:
            mean, lower, upper = bootstrap_results[model_name][metric]
            means.append(mean)
            lowers.append(lower)
            uppers.append(upper)

        means = np.array(means)
        lowers = np.array(lowers)
        uppers = np.array(uppers)
        yerr = np.vstack([means - lowers, uppers - means])

        plt.bar(
            index + i * bar_width,
            means,  # ← 这是Bootstrap均值
            bar_width,
            label=metric,
            color=colors[i % len(colors)],
            yerr=yerr,  # ← 这是Bootstrap置信区间
            capsize=3,
            edgecolor='black',
            linewidth=0.8,
            error_kw={'elinewidth': 1.0, 'capthick': 1.0}
        )

        # 添加数值标签 - 明确显示这是Bootstrap均值
        for j, value in enumerate(means):
            plt.text(
                j + i * bar_width,
                uppers[j] + 0.01,
                f'{value:.3f}',  # ← Bootstrap均值
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90
            )

    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Bootstrap Mean ± 95% CI', fontsize=12)  # ← 明确标注
    plt.title('Model Comparison - Bootstrap Estimates (500 iterations)', fontsize=14)  # ← 明确标注
    plt.xticks(index + bar_width * (len(metrics) - 1) / 2, models, rotation=15)
    plt.ylim(0, 1.05)
    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5)
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    save_path = os.path.join(config['output_dir'], 'bootstrap_model_comparison.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"Bootstrap柱状图已保存到: {save_path}")
    print("📊 图表显示的是Bootstrap估计值（500次重采样的均值）及其95%置信区间")

    return bootstrap_results  # 返回Bootstrap结果用于表格


def train_single_model(model, train_loader, val_loader, device, epochs=10,
                       lr=1e-4, weight_decay=1e-4, patience=5, model_path=None):
    """Train a single backbone model"""

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Get class weights for handling imbalance
    all_train_labels = []
    for batch in train_loader:
        if len(batch) == 3:  # No yellow features
            _, labels, _ = batch
        else:  # With yellow features
            _, _, labels, _ = batch
        all_train_labels.extend(labels.numpy())

    class_weights = compute_class_weights(np.array(all_train_labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Early stopping
    best_val_f1 = 0
    no_improve_epochs = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Training"):
            if len(batch) == 3:  # No yellow features
                inputs, labels, _ = batch
            else:  # With yellow features
                inputs, _, labels, _ = batch

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} Validation"):
                if len(batch) == 3:  # No yellow features
                    inputs, labels, _ = batch
                else:  # With yellow features
                    inputs, _, labels, _ = batch

                inputs, labels = inputs.to(device), labels.to(device)

                _, outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')

        # Update scheduler
        scheduler.step(val_f1)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0

            # Save model
            if model_path:
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")

        # Early stopping
        if no_improve_epochs >= patience:
            print("Early stopping triggered")
            break

    # Load best model if saved
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    return model


def save_comprehensive_results(unified_results, config):
    """保存综合结果到JSON文件"""

    # 转换结果为可序列化的格式
    serializable_results = {}

    for model_name, result in unified_results.items():
        metrics = result['metrics']
        y_true, y_pred, y_prob = result['predictions']

        serializable_results[model_name] = {
            'metrics': {
                'accuracy': float(metrics['accuracy']),
                'f1_macro': float(metrics['f1_macro']),
                'f1_weighted': float(metrics['f1_weighted']),
                'precision_macro': float(metrics['precision_macro']),
                'recall_macro': float(metrics['recall_macro']),
                'sensitivity_mean': float(metrics['sensitivity_mean']),
                'specificity_mean': float(metrics['specificity_mean']),
                'auc': float(metrics['auc']),
                'sensitivity_per_class': [float(x) for x in metrics['sensitivity_per_class']],
                'specificity_per_class': [float(x) for x in metrics['specificity_per_class']],
                'confusion_matrix': metrics['confusion_matrix'].tolist()
            },
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_prob': y_prob.tolist()
            }
        }

    # 保存到JSON文件
    results_path = os.path.join(config['output_dir'], 'comprehensive_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)

    print(f"综合结果已保存到: {results_path}")



def extract_features(model, data_loader, device, include_yellow=True):
    """Extract features from a trained model for all samples in the data loader"""
    model.eval()
    features_list = []
    yellow_list = []
    labels_list = []
    serials_list = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            if include_yellow:
                inputs, yellow, labels, serials = batch
                inputs = inputs.to(device)
                yellow = yellow.to(device)

                # Extract features from backbone
                features, _ = model(inputs)

                # Store features and metadata
                features_list.append(features.cpu())
                yellow_list.append(yellow.cpu())
                labels_list.append(labels)
                serials_list.extend(serials)
            else:
                inputs, labels, serials = batch
                inputs = inputs.to(device)

                # Extract features from backbone
                features, _ = model(inputs)

                # Store features and metadata
                features_list.append(features.cpu())
                labels_list.append(labels)
                serials_list.extend(serials)

    # Concatenate all features
    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    if include_yellow:
        all_yellow = torch.cat(yellow_list, dim=0)
        return all_features, all_yellow, all_labels, serials_list
    else:
        return all_features, all_labels, serials_list


def extract_logits(model, data_loader, device, include_yellow=True):
    model.eval()
    logits_list = []
    yellow_list = []
    labels_list = []
    serials_list = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting logits"):
            if include_yellow:
                inputs, yellow, labels, serials = batch
                inputs = inputs.to(device)
                yellow = yellow.to(device)
                _, logits = model(inputs)
                logits_list.append(logits.cpu())
                yellow_list.append(yellow.cpu())
                labels_list.append(labels)
                serials_list.extend(serials)
            else:
                inputs, labels, serials = batch
                inputs = inputs.to(device)
                _, logits = model(inputs)
                logits_list.append(logits.cpu())
                labels_list.append(labels)
                serials_list.extend(serials)

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    if include_yellow:
        all_yellow = torch.cat(yellow_list, dim=0)
        return all_logits, all_yellow, all_labels, serials_list
    else:
        return all_logits, all_labels, serials_list


def train_ensemble_model(ensemble_model, feature_loaders, train_data, val_data, device,
                         epochs=20, lr=1e-4, weight_decay=1e-4,
                         patience=7, model_path=None, include_yellow=True):

    ensemble_model.to(device)
    optimizer = torch.optim.AdamW(ensemble_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # unpack train data
    if include_yellow:
        train_features, train_yellow, train_labels, _ = train_data
        val_features, val_yellow, val_labels, _ = val_data
    else:
        train_features, train_labels, _ = train_data
        val_features, val_labels, _ = val_data

    class_weights = compute_class_weights(train_labels.numpy()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = 0
    no_improve_epochs = 0

    for epoch in range(epochs):
        ensemble_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels_list = []

        indices = torch.randperm(len(train_labels))
        batch_size = 32

        for i in range(0, len(train_labels), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_features = [feat[batch_indices].to(device) for feat in train_features]
            batch_labels = train_labels[batch_indices].to(device)
            if include_yellow:
                batch_yellow = train_yellow[batch_indices].to(device)
                outputs = ensemble_model(batch_features, batch_yellow)
            else:
                outputs = ensemble_model(batch_features)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_labels.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(batch_labels).sum().item()
            train_total += batch_labels.size(0)
            train_preds.extend(predicted.cpu().numpy())
            train_labels_list.extend(batch_labels.cpu().numpy())

        train_loss = train_loss / train_total if train_total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0
        train_f1 = f1_score(train_labels_list, train_preds, average='macro')

        # ----------- 验证阶段，用val_data -----------
        ensemble_model.eval()
        with torch.no_grad():
            val_batch_features = [feat.to(device) for feat in val_features]
            val_batch_labels = val_labels.to(device)
            if include_yellow:
                val_batch_yellow = val_yellow.to(device)
                outputs = ensemble_model(val_batch_features, val_batch_yellow)
            else:
                outputs = ensemble_model(val_batch_features)
            val_loss = criterion(outputs, val_batch_labels).item()
            _, predicted = outputs.max(1)
            val_correct = predicted.eq(val_batch_labels).sum().item()
            val_total = val_batch_labels.size(0)
            val_preds = predicted.cpu().numpy()
            val_labels_list = val_batch_labels.cpu().numpy()
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_f1 = f1_score(val_labels_list, val_preds, average='macro')

        scheduler.step(val_f1)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            if model_path:
                torch.save(ensemble_model.state_dict(), model_path)
                print(f"Ensemble model saved to {model_path}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")

        if no_improve_epochs >= patience:
            print("Early stopping triggered")
            break

    if model_path and os.path.exists(model_path):
        ensemble_model.load_state_dict(torch.load(model_path))

    return ensemble_model



def evaluate_model(model, test_data, device, class_names=None, is_ensemble=False, feature_loaders=None,
                   include_yellow=True,roc_save_path=None):
    """Evaluate model on test set with detailed metrics"""

    if class_names is None:
        class_names = ['Mild', 'Moderate', 'Severe']

    if is_ensemble and feature_loaders is None:
        raise ValueError("Must provide feature_loaders for ensemble evaluation")

    # Set model to evaluation mode
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        if is_ensemble:
            # For ensemble model, use pre-extracted features
            if include_yellow:
                # Unpack test data
                _, yellow_features, labels, _ = test_data

                # Move features to device
                test_features = []
                for feat in feature_loaders:
                    test_features.append(feat.to(device))

                test_yellow = yellow_features.to(device)
                test_labels = labels.to(device)

                # Forward pass
                outputs = model(test_features, test_yellow)
            else:
                # Unpack test data
                _, labels, _ = test_data

                # Move features to device
                test_features = []
                for feat in feature_loaders:
                    test_features.append(feat.to(device))

                test_labels = labels.to(device)

                # Forward pass
                outputs = model(test_features)

            # Get predictions
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Collect results
            all_preds = preds.cpu().numpy()
            all_labels = test_labels.cpu().numpy()
            all_probs = probs.cpu().numpy()

        else:
            # For single model, iterate through test loader
            for batch in tqdm(test_data, desc="Evaluating"):
                if include_yellow:
                    inputs, yellow, labels, _ = batch
                    inputs = inputs.to(device)
                else:
                    inputs, labels, _ = batch
                    inputs = inputs.to(device)

                labels = labels.to(device)

                # Forward pass
                if hasattr(model, 'forward') and callable(getattr(model, 'forward')):
                    _, outputs = model(inputs)
                else:
                    outputs = model(inputs)

                # Get predictions
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                # Collect results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays if needed
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    # 计算每类的敏感度（Recall）、特异度（Specificity）
    num_classes = len(class_names)
    sensitivity = []
    specificity = []
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FN = conf_matrix[i, :].sum() - TP
        FP = conf_matrix[:, i].sum() - TP
        TN = conf_matrix.sum() - (TP + FP + FN)
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        sensitivity.append(sens)
        specificity.append(spec)

    # 打印
    print("\nPer-class Sensitivity (Recall) & Specificity:")
    table2 = PrettyTable()
    table2.field_names = ["Class", "Sensitivity (Recall)", "Specificity"]
    for i, class_name in enumerate(class_names):
        table2.add_row([class_name, f"{sensitivity[i]:.4f}", f"{specificity[i]:.4f}"])
    print(table2)

    # Calculate ROC curve and AUC
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize

        # Binarize labels for multi-class ROC
        classes = np.unique(all_labels)
        y_bin = label_binarize(all_labels, classes=classes)

        if len(classes) > 2:
            roc_auc = roc_auc_score(y_bin, all_probs, multi_class='ovr', average='weighted')
        else:
            roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception as e:
        print(f"Warning: Could not calculate ROC AUC: {e}")
        roc_auc = 0

    # Print results
    print(f"===== Test Set Evaluation =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")

    # Pretty print confusion matrix
    table = PrettyTable()
    table.field_names = [""] + [f"Pred {name}" for name in class_names]
    for i, row in enumerate(conf_matrix):
        table.add_row([f"True {class_names[i]}"] + list(row))
    print(table)

    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    print("\nPer-class Metrics:")
    table = PrettyTable()
    table.field_names = ["Class", "Precision", "Recall", "F1 Score"]
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            table.add_row([class_name, f"{precision_per_class[i]:.4f}",
                           f"{recall_per_class[i]:.4f}", f"{f1_per_class[i]:.4f}"])
    print(table)
    if roc_save_path is None:
        roc_save_path = "roc_curve.png"
    plot_multiclass_roc(all_labels, all_probs, class_names, save_path=roc_save_path)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "sensitivity_per_class": sensitivity,
        "specificity_per_class": specificity,
        "all_labels": all_labels,
        "all_probs": all_probs
    }


# ========== 7. Main Training Pipeline ==========
def unified_evaluate_all_models(image_models, yellow_model, fusion_model, test_loader, device, config):
    """统一评估所有模型，确保数据一致性"""

    all_results = {}

    # 1. 评估单个图像模型
    for model_name, model in image_models.items():
        print(f"Unified evaluation for {model_name}...")
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                _, outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 转换为numpy数组
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        # 计算统一的指标
        metrics = calculate_unified_metrics(y_true, y_pred, y_prob, config['class_names'])
        all_results[model_name] = {
            'metrics': metrics,
            'predictions': (y_true, y_pred, y_prob)
        }

    # 2. 评估黄疸特征模型
    print("Unified evaluation for Yellow Features...")
    yellow_model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            yellow_features = batch['yellow_features'].to(device)
            labels = batch['label'].to(device)

            outputs = yellow_model(yellow_features)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = calculate_unified_metrics(y_true, y_pred, y_prob, config['class_names'])
    all_results['YellowFeatures'] = {
        'metrics': metrics,
        'predictions': (y_true, y_pred, y_prob)
    }

    # 3. 评估Dynamic Weight Fusion
    print("Unified evaluation for Dynamic Weight Fusion...")
    fusion_model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            yellow_features = batch['yellow_features'].to(device)
            labels = batch['label'].to(device)

            # 获取所有模型的logits
            logits_list = []

            # 图像模型logits
            for model in image_models.values():
                _, logits = model(images)
                logits_list.append(logits)

            # 黄疸模型logits
            yellow_logits = yellow_model(yellow_features)
            logits_list.append(yellow_logits)

            # 堆叠logits
            stacked_logits = torch.stack(logits_list, dim=1)

            # Dynamic fusion
            fused_output, weights = fusion_model(stacked_logits)
            probs = F.softmax(fused_output, dim=1)
            preds = fused_output.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = calculate_unified_metrics(y_true, y_pred, y_prob, config['class_names'])
    all_results['DynamicFusion'] = {
        'metrics': metrics,
        'predictions': (y_true, y_pred, y_prob)
    }

    return all_results

def plot_comprehensive_roc(unified_results, config):
    """绘制所有模型的综合ROC曲线"""

    plt.figure(figsize=(10, 8))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']

    for i, (model_name, result) in enumerate(unified_results.items()):
        y_true, y_pred, y_prob = result['predictions']

        # 计算每个类别的ROC
        n_classes = len(config['class_names'])
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        # 计算macro-average ROC AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for class_idx in range(n_classes):
            fpr[class_idx], tpr[class_idx], _ = roc_curve(y_true_bin[:, class_idx], y_prob[:, class_idx])
            roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

        # 计算macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[j] for j in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for class_idx in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[class_idx], tpr[class_idx])
        mean_tpr /= n_classes

        macro_auc = auc(all_fpr, mean_tpr)

        plt.plot(all_fpr, mean_tpr,
                 label=f'{model_name} (AUC = {macro_auc:.3f})',
                 color=colors[i % len(colors)], linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comprehensive ROC Curves - All Models')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(config['output_dir'], 'comprehensive_roc_curves.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

    print(f"综合ROC曲线已保存到: {save_path}")


def train_and_evaluate_ensemble_layered(config):
    """方案C：分层集成训练和评估（复用已有图像模型）"""
    import json

    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(config['output_dir'], exist_ok=True)
    train_dir = os.path.join(config['dataset_dir'], "train")
    val_dir = os.path.join(config['dataset_dir'], "val")
    test_dir = os.path.join(config['dataset_dir'], "test")

    train_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    val_names = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    test_names = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    print(f"Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")

    # =================== 第一层：加载已有的图像模型 ===================
    print("=" * 50)
    print("第一层：加载已有的图像模型")
    print("=" * 50)

    transforms = {
        'convnext': T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'swin': T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'efficientnet': T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'vit': T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # 加载已有的图像模型
    image_models = {}
    image_loaders = {}  # 用于测试评估

    for model_name in config['models']:
        print(f"\n加载 {model_name} 模型...")

        # 创建模型实例
        if model_name == 'ConvNext':
            image_models[model_name] = ConvNextBackbone(
                model_name='ConvNext', num_classes=3, pretrained=True, freeze_base=config['freeze_base']
            )
        elif model_name == 'Swin':
            image_models[model_name] = SwinTransformerBackbone(
                model_name='Swin', num_classes=3, pretrained=True, freeze_base=config['freeze_base']
            )
        elif model_name == 'EfficientNet':
            image_models[model_name] = EfficientNetBackbone(
                model_name='EfficientNet', num_classes=3, pretrained=True, freeze_base=config['freeze_base']
            )
        elif model_name == 'ViT':
            image_models[model_name] = ViTBackbone(
                model_name='ViT', num_classes=3, pretrained=True, freeze_base=config['freeze_base']
            )

        # 加载已训练的权重
        if config['reuse_image_models'] and model_name in config['existing_model_paths']:
            model_path = config['existing_model_paths'][model_name]
            if os.path.exists(model_path):
                print(f"从 {model_path} 加载已训练的 {model_name} 模型")
                image_models[model_name].load_state_dict(torch.load(model_path, map_location=device))
                image_models[model_name].to(device)
                image_models[model_name].eval()  # 设置为评估模式
                print(f"✓ {model_name} 模型加载成功")
            else:
                print(f"❌ 警告：找不到 {model_name} 的模型文件 {model_path}")
                print(f"将重新训练 {model_name} 模型...")
                # 这里可以添加重新训练的逻辑，或者抛出异常
                raise FileNotFoundError(f"找不到模型文件: {model_path}")
        else:
            print(f"配置为重新训练 {model_name} 模型")
            # 如果需要重新训练，这里添加训练逻辑

        # 创建测试数据集（用于后续评估）
        test_dataset = Jaundice3ClassDataset_ImageOnly(
            test_dir, test_names, split='test',
            bilirubin_csv=config['bilirubin_csv'],
            transform=transforms[model_name.lower()]
        )
        image_loaders[f'{model_name}_test'] = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )

    print(f"\n✓ 成功加载 {len(image_models)} 个图像模型")

    # =================== 第二层：训练黄疸特征分类器 ===================
    print("\n" + "=" * 50)
    print("第二层：训练黄疸特征分类器（只使用黄疸特征）")
    print("=" * 50)

    # 检查是否已有训练好的黄疸分类器
    yellow_model_path = os.path.join(config['output_dir'], 'best_yellow_classifier.pth')

    if os.path.exists(yellow_model_path) and config['use_pretrained']:
        print(f"发现已训练的黄疸分类器: {yellow_model_path}")
        print("加载已训练的黄疸特征分类器...")

        # 创建模型并加载权重
        yellow_model = YellowFeatureClassifier(
            input_dim=8,
            hidden_dim=config['yellow_classifier']['hidden_dim'],
            num_classes=len(config['class_names']),
            dropout=config['yellow_classifier']['dropout']
        ).to(device)

        yellow_model.load_state_dict(torch.load(yellow_model_path, map_location=device))
        yellow_model.eval()
        print("✓ 黄疸特征分类器加载成功")

    else:
        print("训练新的黄疸特征分类器...")

        # 创建黄疸特征数据集
        yellow_train_dataset = YellowFeatureDataset(
            train_dir, train_names,
            bilirubin_csv=config['bilirubin_csv'],
            enable_undersample=True,
            undersample_count=200
        )
        yellow_val_dataset = YellowFeatureDataset(
            val_dir, val_names,
            bilirubin_csv=config['bilirubin_csv']
        )

        yellow_train_loader = DataLoader(
            yellow_train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        yellow_val_loader = DataLoader(
            yellow_val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )

        # 训练黄疸特征分类器
        yellow_model = train_yellow_classifier(
            yellow_train_loader, yellow_val_loader, device, config
        )

    # 评估黄疸特征分类器
    print("\n===== 黄疸特征分类器评估 =====")
    yellow_test_dataset = YellowFeatureDataset(
        test_dir, test_names,
        bilirubin_csv=config['bilirubin_csv']
    )
    yellow_test_loader = DataLoader(
        yellow_test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    evaluate_yellow_model(yellow_model, yellow_test_loader, device, config)

    # =================== 第三层：集成所有预测 ===================
    print("\n" + "=" * 50)
    print("第三层：训练集成权重")
    print("=" * 50)

    # 检查是否已有训练好的集成模型
    ensemble_model_path = os.path.join(config['output_dir'], 'ensemble_final.pth')

    if os.path.exists(ensemble_model_path) and config['use_pretrained']:
        print(f"发现已训练的集成模型: {ensemble_model_path}")
        print("加载已训练的集成模型...")

        ensemble_model = LayeredEnsemble(
            num_models=5,
            num_classes=len(config['class_names']),
            fusion_method=config['ensemble']['fusion_method']
        ).to(device)

        ensemble_model.load_state_dict(torch.load(ensemble_model_path, map_location=device))
        ensemble_model.eval()
        print("✓ 集成模型加载成功")

    else:
        print("训练新的集成模型...")

        # 创建包含图像和黄疸特征的完整数据集（用于集成训练）
        full_train_dataset = Jaundice3ClassDataset_Full(
            train_dir, train_names,
            bilirubin_csv=config['bilirubin_csv'],
            transform=transforms['convnext'],  # 使用统一的变换
            enable_undersample=True,
            undersample_count=200
        )
        full_val_dataset = Jaundice3ClassDataset_Full(
            val_dir, val_names,
            bilirubin_csv=config['bilirubin_csv'],
            transform=transforms['convnext']
        )

        full_train_loader = DataLoader(
            full_train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        full_val_loader = DataLoader(
            full_val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )

        # 训练集成权重
        ensemble_model = train_ensemble_weights(
            image_models, yellow_model, full_train_loader, full_val_loader, device, config
        )

    # =================== 最终评估 ===================
    print("\n" + "=" * 50)
    print("最终评估")
    print("=" * 50)

    # 创建测试数据集
    full_test_dataset = Jaundice3ClassDataset_Full(
        test_dir, test_names,
        bilirubin_csv=config['bilirubin_csv'],
        transform=transforms['convnext']
    )
    full_test_loader = DataLoader(
        full_test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    # 评估最终集成模型
    print("\n===== 最终集成模型评估 =====")
    final_results = evaluate_ensemble_model(
        image_models, yellow_model, ensemble_model, full_test_loader, device, config
    )

    # 保存模型（如果是新训练的）
    if not (os.path.exists(ensemble_model_path) and config['use_pretrained']):
        save_ensemble_model(ensemble_model, config)

    # 生成统一的评估报告
    generate_unified_report(image_models, yellow_model, ensemble_model, final_results, config)

    print("\n方案C分层集成训练完成！")
    return final_results


def save_ensemble_model(ensemble_model, config):
    """只保存集成模型（图像模型已经存在）"""
    ensemble_path = os.path.join(config['output_dir'], 'ensemble_final.pth')
    torch.save(ensemble_model.state_dict(), ensemble_path)
    print(f"✓ 集成模型已保存到: {ensemble_path}")


def generate_unified_report(image_models, yellow_model, ensemble_model, final_results, config):
    """生成统一的评估报告"""

    print("\n" + "=" * 60)
    print("方案C分层集成最终报告")
    print("=" * 60)

    print(f"\n架构总结:")
    print(f"- 第一层: {len(image_models)} 个图像模型 (复用已训练模型)")
    print(f"- 第二层: 1 个黄疸特征分类器 (新训练)")
    print(f"- 第三层: 集成模型 (融合所有预测)")
    print(f"- 融合方法: {config['ensemble']['fusion_method']}")

    print(f"\n复用的图像模型:")
    for model_name in image_models.keys():
        if config['reuse_image_models'] and model_name in config['existing_model_paths']:
            print(f"  ✓ {model_name}: {config['existing_model_paths'][model_name]}")

    print(f"\n最终性能:")
    print(f"- 准确率: {final_results['accuracy']:.4f}")
    print(f"- F1分数: {final_results['f1']:.4f}")

    # 如果使用加权融合，显示权重
    if config['ensemble']['fusion_method'] == 'weighted':
        weights = torch.softmax(ensemble_model.weights, dim=0).cpu().numpy()
        model_names = list(image_models.keys()) + ['YellowFeatures']
        print(f"\n模型权重分配:")
        for name, weight in zip(model_names, weights):
            print(f"  {name}: {weight:.4f}")

    # 保存报告到文件
    report_path = os.path.join(config['output_dir'], 'layered_ensemble_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("方案C分层集成最终报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"架构总结:\n")
        f.write(f"- 第一层: {len(image_models)} 个图像模型 (复用已训练模型)\n")
        f.write(f"- 第二层: 1 个黄疸特征分类器 (新训练)\n")
        f.write(f"- 第三层: 集成模型 (融合所有预测)\n")
        f.write(f"- 融合方法: {config['ensemble']['fusion_method']}\n\n")

        f.write(f"复用的图像模型:\n")
        for model_name in image_models.keys():
            if config['reuse_image_models'] and model_name in config['existing_model_paths']:
                f.write(f"  ✓ {model_name}: {config['existing_model_paths'][model_name]}\n")

        f.write(f"\n最终性能:\n")
        f.write(f"- 准确率: {final_results['accuracy']:.4f}\n")
        f.write(f"- F1分数: {final_results['f1']:.4f}\n\n")

        if config['ensemble']['fusion_method'] == 'weighted':
            weights = torch.softmax(ensemble_model.weights, dim=0).cpu().numpy()
            model_names = list(image_models.keys()) + ['YellowFeatures']
            f.write(f"模型权重分配:\n")
            for name, weight in zip(model_names, weights):
                f.write(f"  {name}: {weight:.4f}\n")

    print(f"\n报告已保存到: {report_path}")


# 添加辅助函数
def train_single_model_image_only(model, train_loader, val_loader, device, epochs=10,
                                  lr=1e-4, weight_decay=1e-4, patience=5, model_path=None):
    """训练只使用图像的单个模型"""

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 获取类别权重
    all_train_labels = []
    for batch in train_loader:
        _, labels, _ = batch
        all_train_labels.extend(labels.numpy())

    class_weights = compute_class_weights(np.array(all_train_labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 早停
    best_val_f1 = 0
    no_improve_epochs = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Training"):
            inputs, labels, _ = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} Validation"):
                inputs, labels, _ = batch
                inputs, labels = inputs.to(device), labels.to(device)

                _, outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels_list, val_preds, average='macro')

        # 更新调度器
        scheduler.step(val_f1)

        # 打印进度
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        # 检查改进
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0

            # 保存模型
            if model_path:
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")

        # 早停
        if no_improve_epochs >= patience:
            print("Early stopping triggered")
            break

    # 加载最佳模型
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    return model


def evaluate_model_image_only(model, test_loader, device, class_names=None, roc_save_path=None):
    """评估只使用图像的模型"""

    if class_names is None:
        class_names = ['Mild', 'Moderate', 'Severe']

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels, _ = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            _, outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"图像模型性能:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 绘制ROC曲线
    if roc_save_path:
        plot_multiclass_roc(all_labels, all_probs, class_names, save_path=roc_save_path)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "all_labels": all_labels,
        "all_preds": all_preds,
        "all_probs": all_probs
    }


def evaluate_yellow_model(model, test_loader, device, config):
    """评估黄疸特征分类器"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            yellow_features = batch['yellow_features'].to(device)
            labels = torch.tensor(batch['label']).to(device)

            outputs = model(yellow_features)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"黄疸特征分类器性能:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=config['class_names']))


def evaluate_ensemble_model(image_models, yellow_model, ensemble_model, test_loader, device, config):
    """评估最终集成模型"""

    # 设置所有模型为评估模式
    for model in image_models.values():
        model.eval()
    yellow_model.eval()
    ensemble_model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            yellow_features = batch['yellow_features'].to(device)
            labels = batch['label'].to(device)

            # 获取所有模型的logits
            logits_list = []

            # 图像模型预测
            for model_name, model in image_models.items():
                _, logits = model(images)
                logits_list.append(logits)

            # 黄疸特征模型预测
            yellow_logits = yellow_model(yellow_features)
            logits_list.append(yellow_logits)

            # 集成预测
            ensemble_logits = ensemble_model(logits_list)
            probabilities = torch.softmax(ensemble_logits, dim=1)
            _, predicted = torch.max(ensemble_logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"最终集成模型性能:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 打印详细分类报告
    print("\n最终分类报告:")
    print(classification_report(all_labels, all_preds, target_names=config['class_names']))

    # 计算AUC
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        print(f"ROC AUC: {auc:.4f}")
    except:
        print("无法计算AUC")

    # 打印模型权重（如果使用加权融合）
    if config['ensemble']['fusion_method'] == 'weighted':
        weights = torch.softmax(ensemble_model.weights, dim=0).cpu().numpy()
        model_names = list(image_models.keys()) + ['YellowFeatures']
        print(f"\n模型权重:")
        for name, weight in zip(model_names, weights):
            print(f"{name}: {weight:.4f}")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_probs': all_probs
    }


# 添加完整数据集类（包含图像和黄疸特征）
class Jaundice3ClassDataset_Full(Dataset):
    """包含图像和黄疸特征的完整数据集类（用于集成训练）"""

    def __init__(self, root_dir, patient_names, split='train',
                 bilirubin_csv=None, transform=None,
                 enable_undersample=False, undersample_count=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_names = ['mild', 'moderate', 'severe']
        self.class_to_idx = {name: i for i, name in enumerate(self.label_names)}

        # 初始化特征提取器
        self.ffe = FacialFeatureExtractor(use_mediapipe=True)

        self.samples = []

        # 读取胆红素数据
        assert bilirubin_csv is not None, "Must provide bilirubin csv file path"
        df = pd.read_excel(bilirubin_csv)
        df.columns = df.columns.str.strip()
        self.serial2bil = dict(zip(df['序号'], df['26、总胆红素值（umol/L）']))

        for folder_name in patient_names:
            serial = extract_serial_number(folder_name)
            if serial is None or serial not in self.serial2bil:
                print(f"[Warning] {folder_name} not matched to table serial number, skipping")
                continue

            bil = float(self.serial2bil[serial])
            # 直接三分类
            if bil < 171:
                label_name = 'mild'
            elif bil < 342:
                label_name = 'moderate'
            else:
                label_name = 'severe'
            label = self.class_to_idx[label_name]

            patient_folder = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(patient_folder):
                continue

            imgs = [f for f in os.listdir(patient_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not imgs:
                print(f"[Warning] No images in {patient_folder}, skipping")
                continue

            for img_name in imgs:
                img_path = os.path.join(patient_folder, img_name)
                self.samples.append((img_path, label, serial))

        # 下采样处理
        if self.split == 'train' and enable_undersample:
            print("Applying undersampling to full dataset...")
            label_to_samples = defaultdict(list)
            for sample in self.samples:
                label_to_samples[sample[1]].append(sample)
            if undersample_count is None:
                undersample_count = min(len(s) for s in label_to_samples.values())
            print(f"Undersample count per class: {undersample_count}")
            new_samples = []
            for label, group in label_to_samples.items():
                if len(group) >= undersample_count:
                    new_samples.extend(random.sample(group, undersample_count))
                else:
                    new_samples.extend(group)
            self.samples = new_samples
            print(f"After undersampling: {Counter([s[1] for s in self.samples])}")

        print(f"Loaded {len(self.samples)} full samples. Class distribution:")
        print(Counter([s[1] for s in self.samples]))

    def extract_yellow_features(self, image):
        """从PIL图像中提取黄疸特征"""
        try:
            # 提取眼部区域
            left_eye, right_eye = self.ffe.extract_sclera_patches(image)

            # 确保区域提取成功
            if left_eye is None or right_eye is None:
                # 回退到直接裁剪
                w, h = image.size
                left_eye = image.crop((0, 0, w // 4, h // 4)) if left_eye is None else left_eye
                right_eye = image.crop((3 * w // 4, 0, w, h // 4)) if right_eye is None else right_eye

            # 提取黄疸指标
            left_eye_yellow = extract_yellow_metrics(left_eye)
            right_eye_yellow = extract_yellow_metrics(right_eye)

            # 合并特征 (每个眼部4个特征，总共8个)
            features = np.concatenate([left_eye_yellow, right_eye_yellow])
            return features

        except Exception as e:
            print(f"特征提取错误: {e}")
            return np.zeros(8)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_path, label, serial = self.samples[idx]
            img = Image.open(img_path).convert('RGB')

            # Apply data augmentation for training
            if self.split == 'train':
                np_img = np.array(img)
                np_img = color_sensitive_augmentation(np_img)
                img = Image.fromarray(np_img)

            # 提取黄疸特征（在变换之前）
            yellow_features = self.extract_yellow_features(img)

            # Apply transform if provided
            if self.transform:
                img_tensor = self.transform(img)
            else:
                # Default transform
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_tensor = transform(img)

            return {
                'image': img_tensor,
                'yellow_features': torch.FloatTensor(yellow_features),
                'label': torch.tensor(label),
                'serial': serial
            }

        except Exception as e:
            print(f"Cannot process sample {idx}, error: {e}")
            return self.__getitem__(max(0, idx - 1))


def save_all_models(image_models, yellow_model, ensemble_model, config):
    """保存所有训练好的模型"""

    # 保存图像模型
    for model_name, model in image_models.items():
        model_path = os.path.join(config['output_dir'], f"{model_name.lower()}_final.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved {model_name} to {model_path}")

    # 保存黄疸特征模型
    yellow_path = os.path.join(config['output_dir'], 'yellow_classifier_final.pth')
    torch.save(yellow_model.state_dict(), yellow_path)
    print(f"Saved yellow classifier to {yellow_path}")

    # 保存集成模型
    ensemble_path = os.path.join(config['output_dir'], 'ensemble_final.pth')
    torch.save(ensemble_model.state_dict(), ensemble_path)
    print(f"Saved ensemble model to {ensemble_path}")


def generate_unified_report(image_models, yellow_model, ensemble_model, final_results, config):
    """生成统一的评估报告"""

    print("\n" + "=" * 60)
    print("方案C分层集成最终报告")
    print("=" * 60)

    print(f"\n架构总结:")
    print(f"- 第一层: {len(image_models)} 个图像模型 (只使用图像)")
    print(f"- 第二层: 1 个黄疸特征分类器 (只使用黄疸特征)")
    print(f"- 第三层: 集成模型 (融合所有预测)")
    print(f"- 融合方法: {config['ensemble']['fusion_method']}")

    print(f"\n最终性能:")
    print(f"- 准确率: {final_results['accuracy']:.4f}")
    print(f"- F1分数: {final_results['f1']:.4f}")

    # 如果使用加权融合，显示权重
    if config['ensemble']['fusion_method'] == 'weighted':
        weights = torch.softmax(ensemble_model.weights, dim=0).cpu().numpy()
        model_names = list(image_models.keys()) + ['YellowFeatures']
        print(f"\n模型权重分配:")
        for name, weight in zip(model_names, weights):
            print(f"  {name}: {weight:.4f}")

    print(f"\n训练配置:")
    print(f"- 图像模型训练轮数: {config['backbone_epochs']}")
    print(f"- 黄疸分类器训练轮数: {config['yellow_classifier']['epochs']}")
    print(f"- 集成权重训练轮数: {config['ensemble']['weights_epochs']}")
    print(f"- 批次大小: {config['batch_size']}")

    # 保存报告到文件
    report_path = os.path.join(config['output_dir'], 'layered_ensemble_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("方案C分层集成最终报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"架构总结:\n")
        f.write(f"- 第一层: {len(image_models)} 个图像模型 (只使用图像)\n")
        f.write(f"- 第二层: 1 个黄疸特征分类器 (只使用黄疸特征)\n")
        f.write(f"- 第三层: 集成模型 (融合所有预测)\n")
        f.write(f"- 融合方法: {config['ensemble']['fusion_method']}\n\n")
        f.write(f"最终性能:\n")
        f.write(f"- 准确率: {final_results['accuracy']:.4f}\n")
        f.write(f"- F1分数: {final_results['f1']:.4f}\n\n")

        if config['ensemble']['fusion_method'] == 'weighted':
            weights = torch.softmax(ensemble_model.weights, dim=0).cpu().numpy()
            model_names = list(image_models.keys()) + ['YellowFeatures']
            f.write(f"模型权重分配:\n")
            for name, weight in zip(model_names, weights):
                f.write(f"  {name}: {weight:.4f}\n")

    print(f"\n报告已保存到: {report_path}")



# 更新配置
config = {
    'seed': 42,
    'dataset_dir': r"C:\Users\MichaelY\Desktop\jaundice_dataset\arcface_dataset",
    'bilirubin_csv': r"C:\Users\MichaelY\Documents\WeChat Files\wxid_sckyac3nu7h521\FileStorage\File\2025-05\213008980_按序号_肝炎患者补充问卷-医护_313_313.xlsx",
    'output_dir': r"C:\Users\MichaelY\Desktop\jaundice3cls",
    'models': ['ConvNext', 'Swin', 'EfficientNet', 'ViT'],
    'features': ['yellow'],
    'batch_size': 8,
    'num_workers': 0,
    'freeze_base': True,
    'use_pretrained': True,

    # 复用已有的图像模型
    'reuse_image_models': True,
    'existing_model_paths': {
        'ConvNext': r"C:\Users\MichaelY\Desktop\jaundice3cls\convnext_best.pth",
        'Swin': r"C:\Users\MichaelY\Desktop\jaundice3cls\swin_best.pth",
        'EfficientNet': r"C:\Users\MichaelY\Desktop\jaundice3cls\efficientnet_best.pth",
        'ViT': r"C:\Users\MichaelY\Desktop\jaundice3cls\vit_best.pth"
    },

    # 黄疸特征分类器配置
    'yellow_classifier': {
        'hidden_dim': 64,
        'dropout': 0.3,
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 1e-4
    },

    # 集成配置 - 使用Self Attention融合
    'ensemble': {
        'fusion_method': 'dynamic_weight',  # 使用真正的Self Attention
        'lr': 0.001,
        'weight_decay': 1e-4,
        'epochs': 100,
        'patience': 15,  # 早停耐心值
        'min_delta': 0.001,  # 早停最小改进
        'hidden_dim': 64
    },

    'class_names': ['Mild', 'Moderate', 'Severe']
}


def check_dependencies():
    """检查必要的依赖和文件"""

    # 检查模型文件是否存在
    missing_models = []
    for model_name, path in config['existing_model_paths'].items():
        if not os.path.exists(path):
            missing_models.append(f"{model_name}: {path}")

    if missing_models:
        print("❌ 缺失以下模型文件:")
        for missing in missing_models:
            print(f"  - {missing}")
        return False

    # 检查数据集目录
    if not os.path.exists(config['dataset_dir']):
        print(f"❌ 数据集目录不存在: {config['dataset_dir']}")
        return False

    # 检查胆红素CSV文件
    if not os.path.exists(config['bilirubin_csv']):
        print(f"❌ 胆红素CSV文件不存在: {config['bilirubin_csv']}")
        return False

    print("✅ 所有依赖检查通过")
    return True


if __name__ == "__main__":
    # 检查依赖
    if not check_dependencies():
        print("请先准备好所需的模型文件和数据")
        exit(1)

    print("🚀 开始Dynamic Weight Fusion完整训练和评估...")

    try:
        # 使用完整的训练和评估函数
        unified_results, bootstrap_results = train_and_evaluate_dynamic_fusion_comprehensive(config)

        print("\n✅ 所有任务完成!")

    except Exception as e:
        print(f"\n❌ 运行过程中出现错误: {e}")
        import traceback

        traceback.print_exc()

