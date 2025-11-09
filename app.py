"""
Flask后端服务器 - 黄疸严重程度分类系统（视频版本）- 中英双语版
Flask Backend Server - Jaundice Severity Classification System (Video Version) - Bilingual
使用Dynamic Weight Fusion模型进行预测
Uses Dynamic Weight Fusion model for prediction
支持视频输入和多帧聚合
Supports video input and multi-frame aggregation
"""

import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from werkzeug.utils import secure_filename
import tempfile
import traceback
from segment_anything import sam_model_registry, SamPredictor
import re

app = Flask(__name__)
CORS(app)  # 允许跨域请求 / Enable CORS

# 配置 / Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# 全局变量存储模型 / Global variables for models
models = {}
device = None
transform = None
ffe = None
sam_predictor = None

# ============ 多语言支持 / Multilingual Support ============

class MessageManager:
    """消息管理器 / Message Manager"""

    def __init__(self):
        self.messages = {
            'zh': {
                # 系统状态 / System Status
                'api_running': '黄疸分类器API正在运行（支持视频和图片）',
                'models_loaded': '模型已加载',
                'system_healthy': '系统健康',
                'server_starting': '🚀 启动黄疸分类器服务器（视频版本）...',
                'server_success': '✅ 服务器启动成功!',
                'server_failed': '❌ 服务器启动失败',
                'models_loading_success': '✅ 所有模型加载成功!',
                'models_loading_failed': '❌ 模型加载失败',

                # 错误信息 / Error Messages
                'no_file_uploaded': '没有上传文件',
                'no_file_selected': '没有选择文件',
                'unsupported_format': '不支持的文件格式',
                'prediction_failed': '预测失败',
                'request_failed': '请求处理失败',
                'video_processing_failed': '视频处理失败',
                'no_valid_frames': '未能从视频中提取到有效帧',
                'feature_extraction_failed': '特征提取失败',

                # 处理过程 / Processing
                'processing_frame': '处理帧',
                'predicting_frame': '预测第',
                'extracting_frames': '成功提取',
                'frames_unit': '帧',
                'video_processing_start': '开始处理视频',

                # 分类结果 / Classification Results
                'mild': '轻度',
                'moderate': '中度',
                'severe': '重度',

                # API端点描述 / API Endpoints
                'endpoint_image': '图片分类',
                'endpoint_video': '视频分类',
                'endpoint_health': '健康检查',

                # 设备信息 / Device Info
                'using_device': '使用设备',
                'device_not_initialized': '设备未初始化',

                # 模型加载 / Model Loading
                'loading_convnext': '加载ConvNext模型...',
                'loading_swin': '加载Swin模型...',
                'loading_efficientnet': '加载EfficientNet模型...',
                'loading_vit': '加载ViT模型...',
                'loading_yellow_classifier': '加载黄疸特征分类器...',
                'loading_dynamic_fusion': '加载Dynamic Fusion模型...',
                'initializing_sam': '初始化SAM模型...',
            },
            'en': {
                # System Status
                'api_running': 'Jaundice Classifier API is running (supports video and images)',
                'models_loaded': 'Models loaded',
                'system_healthy': 'System healthy',
                'server_starting': '🚀 Starting Jaundice Classifier Server (Video Version)...',
                'server_success': '✅ Server started successfully!',
                'server_failed': '❌ Server startup failed',
                'models_loading_success': '✅ All models loaded successfully!',
                'models_loading_failed': '❌ Model loading failed',

                # Error Messages
                'no_file_uploaded': 'No file uploaded',
                'no_file_selected': 'No file selected',
                'unsupported_format': 'Unsupported file format',
                'prediction_failed': 'Prediction failed',
                'request_failed': 'Request processing failed',
                'video_processing_failed': 'Video processing failed',
                'no_valid_frames': 'Failed to extract valid frames from video',
                'feature_extraction_failed': 'Feature extraction failed',

                # Processing
                'processing_frame': 'Processing frame',
                'predicting_frame': 'Predicting frame',
                'extracting_frames': 'Successfully extracted',
                'frames_unit': 'frames',
                'video_processing_start': 'Starting video processing',

                # Classification Results
                'mild': 'mild',
                'moderate': 'moderate',
                'severe': 'severe',

                # API Endpoints
                'endpoint_image': 'Image classification',
                'endpoint_video': 'Video classification',
                'endpoint_health': 'Health check',

                # Device Info
                'using_device': 'Using device',
                'device_not_initialized': 'Device not initialized',

                # Model Loading
                'loading_convnext': 'Loading ConvNext model...',
                'loading_swin': 'Loading Swin model...',
                'loading_efficientnet': 'Loading EfficientNet model...',
                'loading_vit': 'Loading ViT model...',
                'loading_yellow_classifier': 'Loading Yellow Feature Classifier...',
                'loading_dynamic_fusion': 'Loading Dynamic Fusion model...',
                'initializing_sam': 'Initializing SAM model...',
            }
        }

    def get_message(self, key, lang='zh'):
        """获取指定语言的消息 / Get message in specified language"""
        if lang not in self.messages:
            lang = 'zh'  # 默认中文 / Default to Chinese
        return self.messages[lang].get(key, key)

    def get_bilingual_message(self, key):
        """获取双语消息 / Get bilingual message"""
        return {
            'zh': self.get_message(key, 'zh'),
            'en': self.get_message(key, 'en')
        }

# 创建消息管理器实例 / Create message manager instance
msg_manager = MessageManager()

def get_language_from_request():
    """从请求中获取语言设置 / Get language setting from request"""
    # 优先从表单参数获取 / Priority from form parameters
    lang = request.form.get('language', '')
    if not lang:
        # 从URL参数获取 / Get from URL parameters
        lang = request.args.get('language', '')
    if not lang:
        # 从请求头获取 / Get from request headers
        accept_lang = request.headers.get('Accept-Language', '')
        if 'en' in accept_lang.lower():
            lang = 'en'
        else:
            lang = 'zh'

    return 'en' if lang.lower() in ['en', 'english'] else 'zh'

def create_bilingual_response(data, lang='zh'):
    """创建双语响应 / Create bilingual response"""
    if lang == 'en':
        return data
    else:
        # 为中文用户添加英文对照 / Add English reference for Chinese users
        if isinstance(data, dict) and 'predicted_class' in data:
            class_mapping = {
                'mild': {'zh': '轻度', 'en': 'mild'},
                'moderate': {'zh': '中度', 'en': 'moderate'},
                'severe': {'zh': '重度', 'en': 'severe'}
            }

            original_class = data['predicted_class']
            if original_class in class_mapping:
                data['predicted_class_bilingual'] = class_mapping[original_class]

            # 添加双语框架级别预测 / Add bilingual frame-level predictions
            if 'frame_predictions' in data:
                for frame_pred in data['frame_predictions']:
                    orig_class = frame_pred['predicted_class']
                    if orig_class in class_mapping:
                        frame_pred['predicted_class_bilingual'] = class_mapping[orig_class]

    return data

# ============ 从原代码复制必要的类定义 / Copy necessary class definitions from original code ============

class FacialFeatureExtractor:
    """面部特征提取器 / Facial Feature Extractor"""

    def __init__(self, use_mediapipe=True):
        self.use_mediapipe = use_mediapipe
        if use_mediapipe:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

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
        """提取左右眼部区域 / Extract left and right eye regions"""
        landmarks, img_shape = self.get_face_landmarks(image)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if landmarks is None:
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


def extract_yellow_metrics(eye_patch):
    """提取黄疸特征 / Extract jaundice features"""
    img = np.array(eye_patch)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # 黄色检测 / Yellow detection
    yellow_hsv = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([40, 255, 255]))

    # LAB颜色空间的b通道（黄-蓝轴） / LAB color space b channel (yellow-blue axis)
    b_channel = lab[:, :, 2]
    yellow_lab = b_channel > 128

    features = [
        np.mean(yellow_hsv > 0),
        np.mean(yellow_lab),
        np.std(b_channel),
        np.percentile(b_channel, 90)
    ]

    return np.array(features)

class ConvNextBackbone(nn.Module):
    def __init__(self, model_name, num_classes=3, pretrained=True, freeze_base=True):
        super().__init__()
        import torchvision
        self.model_name = model_name
        self.num_classes = num_classes
        weights = 'DEFAULT' if pretrained else None
        self.backbone = torchvision.models.convnext_base(weights=weights)
        self.feature_dim = 1024
        self.original_classifier = self.backbone.classifier
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return features, logits

class SwinTransformerBackbone(nn.Module):
    def __init__(self, model_name, num_classes=3, pretrained=True, freeze_base=True):
        super().__init__()
        import torchvision
        self.model_name = model_name
        self.num_classes = num_classes
        weights = 'DEFAULT' if pretrained else None
        self.backbone = torchvision.models.swin_v2_b(weights=weights)
        self.feature_dim = 1024
        self.original_head = self.backbone.head
        self.backbone.head = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return features, logits

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name, num_classes=3, pretrained=True, freeze_base=True):
        super().__init__()
        import torchvision
        self.model_name = model_name
        self.num_classes = num_classes
        weights = 'DEFAULT' if pretrained else None
        self.backbone = torchvision.models.efficientnet_v2_l(weights=weights)
        self.feature_dim = 1280
        self.original_classifier = self.backbone.classifier
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return features, logits

class ViTBackbone(nn.Module):
    def __init__(self, model_name, num_classes=3, pretrained=True, freeze_base=True):
        super().__init__()
        import torchvision
        self.model_name = model_name
        self.num_classes = num_classes
        weights = 'DEFAULT' if pretrained else None
        self.backbone = torchvision.models.vit_b_16(weights=weights)
        self.feature_dim = 768
        self.original_head = self.backbone.heads
        self.backbone.heads = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return features, logits

class YellowFeatureClassifier(nn.Module):
    """黄疸特征分类器 / Jaundice Feature Classifier"""

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

class DynamicWeightFusion(nn.Module):
    """动态权重融合模型 / Dynamic Weight Fusion Model"""

    def __init__(self, num_models=5, num_classes=3, hidden_dim=64):
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes

        # 权重生成网络 / Weight generation network
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
            nn.Softmax(dim=-1)
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, logits):
        batch_size = logits.size(0)
        flattened = logits.view(batch_size, -1)
        weights = self.weight_generator(flattened)
        weights = F.softmax(weights / self.temperature, dim=-1)
        weighted_logits = torch.sum(logits * weights.unsqueeze(-1), dim=1)
        return weighted_logits, weights


# ============ 视频处理相关函数 / Video Processing Functions ============

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def init_sam(model_type='vit_h', checkpoint_path='E:/sam_vit_h_4b8939.pth'):
    """初始化SAM模型 / Initialize SAM model"""
    global device
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def process_video_frame(frame, predictor, face_mesh):
    """处理单个视频帧，返回分割后的人脸图像 / Process single video frame, return segmented face image"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    h, w, _ = frame.shape
    keypoints = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark])

    # 选择关键点进行SAM分割 / Select key points for SAM segmentation
    selected_indices = [1, 33, 61, 199, 263, 291]
    input_points = keypoints[selected_indices]
    input_labels = np.ones(len(input_points), dtype=int)

    predictor.set_image(frame_rgb)
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )

    mask = masks[0]
    masked_frame = frame_rgb.copy()
    masked_frame[~mask] = [0, 0, 0]

    return Image.fromarray(masked_frame)


def preprocess_frame(image):
    """预处理单帧图像 / Preprocess single frame image"""
    # 使用MediaPipe进行人脸检测 / Use MediaPipe for face detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    img = np.array(image)
    results = face_mesh.process(img)

    if not results.multi_face_landmarks:
        face_mesh.close()
        return image

    landmarks = results.multi_face_landmarks[0].landmark

    # CLAHE光照均衡化 / CLAHE illumination equalization
    def apply_clahe(img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    img = apply_clahe(img)

    # 灰度世界颜色校正 / Gray world color correction
    def gray_world_balance(img):
        result = img.copy().astype(np.float32)
        avg_b, avg_g, avg_r = np.mean(result, axis=(0,1))
        avg_gray = (avg_b + avg_g + avg_r) / 3
        result[:,:,0] *= avg_gray / avg_b
        result[:,:,1] *= avg_gray / avg_g
        result[:,:,2] *= avg_gray / avg_r
        return np.clip(result, 0, 255).astype(np.uint8)

    img = gray_world_balance(img)

    # 眼睛区域增强 / Eye region enhancement
    def enhance_eye_region(img, landmarks):
        h, w = img.shape[:2]
        left_eye_idx = [33, 133, 160, 158, 159, 144, 145, 153]
        right_eye_idx = [362, 263, 387, 385, 386, 373, 374, 380]

        mask = np.zeros((h, w), dtype=np.uint8)
        for eye_idx in [left_eye_idx, right_eye_idx]:
            points = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in eye_idx])
            cv2.fillPoly(mask, [points], 255)

        enhanced = img.copy()
        alpha, beta = 1.3, 15
        eye_region = cv2.bitwise_and(enhanced, enhanced, mask=mask)
        eye_region = cv2.convertScaleAbs(eye_region, alpha=alpha, beta=beta)
        inv_mask = cv2.bitwise_not(mask)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=inv_mask)
        enhanced = cv2.add(enhanced, eye_region)

        return enhanced

    img = enhance_eye_region(img, landmarks)
    face_mesh.close()

    return Image.fromarray(img)


def extract_and_process_video_frames(video_path, sam_predictor, max_frames=12, lang='zh'):
    """从视频中提取和处理帧 / Extract and process frames from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        error_msg = f"无法打开视频: {video_path}" if lang == 'zh' else f"Cannot open video: {video_path}"
        raise ValueError(error_msg)

    # 初始化MediaPipe / Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames)

    processed_frames = []
    frame_count = 0
    saved_count = 0

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # 使用SAM进行人脸分割 / Use SAM for face segmentation
            segmented_frame = process_video_frame(frame, sam_predictor, face_mesh)

            if segmented_frame is not None:
                # 预处理分割后的帧 / Preprocess segmented frame
                preprocessed_frame = preprocess_frame(segmented_frame)
                processed_frames.append(preprocessed_frame)
                saved_count += 1

                processing_msg = msg_manager.get_message('processing_frame', lang)
                print(f"{processing_msg} {saved_count}/{max_frames}")

        frame_count += 1

    cap.release()
    face_mesh.close()

    return processed_frames


def predict_single_frame(image, models, device, transform, ffe):
    """对单帧图像进行预测 / Predict single frame image"""
    # 预处理图像 / Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 提取黄疸特征 / Extract jaundice features
    yellow_features = extract_yellow_features_from_image(image).to(device)

    # 获取所有模型的logits / Get logits from all models
    logits_list = []

    with torch.no_grad():
        # 图像模型预测 / Image model prediction
        for model_name in ['ConvNext', 'Swin', 'EfficientNet', 'ViT']:
            _, logits = models[model_name](image_tensor)
            logits_list.append(logits)

        # 黄疸特征模型预测 / Jaundice feature model prediction
        yellow_logits = models['YellowFeatures'](yellow_features)
        logits_list.append(yellow_logits)

        # 堆叠logits / Stack logits
        stacked_logits = torch.stack(logits_list, dim=1)

        # Dynamic Fusion
        fused_output, weights = models['DynamicFusion'](stacked_logits)

        # 计算概率 / Calculate probabilities
        probabilities = F.softmax(fused_output, dim=1)

    return probabilities.cpu().numpy()[0], weights.cpu().numpy()[0]


def aggregate_frame_predictions(frame_probabilities, method='mean_prob'):
    """聚合多帧预测结果 / Aggregate multi-frame prediction results"""
    frame_probs = np.array(frame_probabilities)

    if method == 'voting':
        # 硬投票 / Hard voting
        frame_preds = np.argmax(frame_probs, axis=1)
        vote_counts = np.bincount(frame_preds, minlength=3)
        final_pred = np.argmax(vote_counts)
        final_prob = np.mean(frame_probs, axis=0)

    elif method == 'mean_prob':
        # 软投票（平均概率） / Soft voting (average probability)
        final_prob = np.mean(frame_probs, axis=0)
        final_pred = np.argmax(final_prob)

    elif method == 'max_prob':
        # 选择最大置信度的预测 / Select prediction with maximum confidence
        max_probs = np.max(frame_probs, axis=1)
        best_frame_idx = np.argmax(max_probs)
        final_prob = frame_probs[best_frame_idx]
        final_pred = np.argmax(final_prob)

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return final_pred, final_prob


def extract_yellow_features_from_image(image):
    """从图像提取黄疸特征 / Extract jaundice features from image"""
    try:
        left_eye, right_eye = ffe.extract_sclera_patches(image)
        left_features = extract_yellow_metrics(left_eye)
        right_features = extract_yellow_metrics(right_eye)
        features = np.concatenate([left_features, right_features])
        return torch.FloatTensor(features).unsqueeze(0)
    except Exception as e:
        print(f"特征提取失败 / Feature extraction failed: {e}")
        return torch.zeros(1, 8)


# ============ 初始化函数 / Initialization Functions ============

def initialize_models():
    """初始化所有模型 / Initialize all models"""
    global models, device, transform, ffe, sam_predictor

    # 设置设备 / Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_msg = msg_manager.get_message('using_device', 'zh')
    print(f"{device_msg} / Using device: {device}")

    # 设置图像变换 / Set image transformations
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 初始化特征提取器 / Initialize feature extractor
    ffe = FacialFeatureExtractor(use_mediapipe=True)

    model_paths = {
        'ConvNext': 'convnext_best.pth',
        'Swin': 'swin_best.pth',
        'EfficientNet': 'efficientnet_best.pth',
        'ViT': 'vit_best.pth',
        'YellowFeatures': 'best_yellow_classifier.pth',
        'DynamicFusion': 'best_dynamic_fusion.pth'
    }

    try:
        # 加载图像模型 / Load image models
        print(msg_manager.get_message('loading_convnext', 'zh'))
        print(msg_manager.get_message('loading_convnext', 'en'))
        models['ConvNext'] = ConvNextBackbone('ConvNext', num_classes=3, pretrained=False)
        models['ConvNext'].load_state_dict(torch.load(model_paths['ConvNext'], map_location=device))
        models['ConvNext'].to(device).eval()

        print(msg_manager.get_message('loading_swin', 'zh'))
        print(msg_manager.get_message('loading_swin', 'en'))
        models['Swin'] = SwinTransformerBackbone('Swin', num_classes=3, pretrained=False)
        models['Swin'].load_state_dict(torch.load(model_paths['Swin'], map_location=device))
        models['Swin'].to(device).eval()

        print(msg_manager.get_message('loading_efficientnet', 'zh'))
        print(msg_manager.get_message('loading_efficientnet', 'en'))
        models['EfficientNet'] = EfficientNetBackbone('EfficientNet', num_classes=3, pretrained=False)
        models['EfficientNet'].load_state_dict(torch.load(model_paths['EfficientNet'], map_location=device))
        models['EfficientNet'].to(device).eval()

        print(msg_manager.get_message('loading_vit', 'zh'))
        print(msg_manager.get_message('loading_vit', 'en'))
        models['ViT'] = ViTBackbone('ViT', num_classes=3, pretrained=False)
        models['ViT'].load_state_dict(torch.load(model_paths['ViT'], map_location=device))
        models['ViT'].to(device).eval()

        # 加载黄疸特征分类器 / Load jaundice feature classifier
        print(msg_manager.get_message('loading_yellow_classifier', 'zh'))
        print(msg_manager.get_message('loading_yellow_classifier', 'en'))
        models['YellowFeatures'] = YellowFeatureClassifier(input_dim=8, hidden_dim=64, num_classes=3)
        models['YellowFeatures'].load_state_dict(torch.load(model_paths['YellowFeatures'], map_location=device))
        models['YellowFeatures'].to(device).eval()

        # 加载Dynamic Fusion模型 / Load Dynamic Fusion model
        print(msg_manager.get_message('loading_dynamic_fusion', 'zh'))
        print(msg_manager.get_message('loading_dynamic_fusion', 'en'))
        models['DynamicFusion'] = DynamicWeightFusion(num_models=5, num_classes=3, hidden_dim=64)
        models['DynamicFusion'].load_state_dict(torch.load(model_paths['DynamicFusion'], map_location=device))
        models['DynamicFusion'].to(device).eval()

        # 初始化SAM模型 / Initialize SAM model
        print(msg_manager.get_message('initializing_sam', 'zh'))
        print(msg_manager.get_message('initializing_sam', 'en'))
        sam_predictor = init_sam(model_type='vit_h', checkpoint_path='E:/sam_vit_h_4b8939.pth')

        print(msg_manager.get_message('models_loading_success', 'zh'))
        print(msg_manager.get_message('models_loading_success', 'en'))

    except Exception as e:
        error_msg_zh = msg_manager.get_message('models_loading_failed', 'zh')
        error_msg_en = msg_manager.get_message('models_loading_failed', 'en')
        print(f"{error_msg_zh} / {error_msg_en}: {e}")
        traceback.print_exc()
        raise


# ============ Flask路由 / Flask Routes ============

@app.route('/')
def index():
    lang = get_language_from_request()

    if lang == 'zh':
        return jsonify({
            'status': 'ok',
            'message': msg_manager.get_message('api_running', 'zh'),
            'message_en': msg_manager.get_message('api_running', 'en'),
            'models': list(models.keys()) if models else [],
            'supported_formats': {
                'video': list(ALLOWED_VIDEO_EXTENSIONS),
                'image': list(ALLOWED_IMAGE_EXTENSIONS)
            },
            'language': 'zh'
        })
    else:
        return jsonify({
            'status': 'ok',
            'message': msg_manager.get_message('api_running', 'en'),
            'message_zh': msg_manager.get_message('api_running', 'zh'),
            'models': list(models.keys()) if models else [],
            'supported_formats': {
                'video': list(ALLOWED_VIDEO_EXTENSIONS),
                'image': list(ALLOWED_IMAGE_EXTENSIONS)
            },
            'language': 'en'
        })


@app.route('/predict_image', methods=['POST'])
def predict_image():
    """处理图片预测请求 / Handle image prediction requests"""
    lang = get_language_from_request()

    try:
        if 'image' not in request.files:
            error_msg = msg_manager.get_message('no_file_uploaded', lang)
            return jsonify({'error': error_msg}), 400

        file = request.files['image']
        if file.filename == '':
            error_msg = msg_manager.get_message('no_file_selected', lang)
            return jsonify({'error': error_msg}), 400

        if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # 预处理图像 / Preprocess image
                pil_image = Image.open(filepath).convert('RGB')
                preprocessed_image = preprocess_frame(pil_image)

                # 进行预测 / Make prediction
                probabilities, model_weights = predict_single_frame(
                    preprocessed_image, models, device, transform, ffe
                )

                # 准备结果 / Prepare results
                class_names_en = ['mild', 'moderate', 'severe']
                class_names_zh = [msg_manager.get_message(name, 'zh') for name in class_names_en]

                predicted_idx = np.argmax(probabilities)
                predicted_class = class_names_zh[predicted_idx] if lang == 'zh' else class_names_en[predicted_idx]

                # 准备模型权重信息 / Prepare model weights information
                model_names = ['ConvNext', 'Swin', 'EfficientNet', 'ViT', 'YellowFeatures']
                model_weights_dict = {name: float(weight) for name, weight in zip(model_names, model_weights)}

                result = {
                    'predicted_class': predicted_class,
                    'probabilities': {
                        msg_manager.get_message('mild', lang): float(probabilities[0]),
                        msg_manager.get_message('moderate', lang): float(probabilities[1]),
                        msg_manager.get_message('severe', lang): float(probabilities[2])
                    },
                    'model_weights': model_weights_dict,
                    'language': lang
                }

                # 添加双语信息 / Add bilingual information
                if lang == 'zh':
                    result['probabilities_en'] = {
                        'mild': float(probabilities[0]),
                        'moderate': float(probabilities[1]),
                        'severe': float(probabilities[2])
                    }
                    result['predicted_class_en'] = class_names_en[predicted_idx]

                os.remove(filepath)
                return jsonify(result)

            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                error_msg = msg_manager.get_message('prediction_failed', lang)
                print(f"{error_msg}: {e}")
                traceback.print_exc()
                return jsonify({'error': f'{error_msg}: {str(e)}'}), 500

        error_msg = msg_manager.get_message('unsupported_format', lang)
        return jsonify({'error': error_msg}), 400

    except Exception as e:
        error_msg = msg_manager.get_message('request_failed', lang)
        print(f"{error_msg}: {e}")
        traceback.print_exc()
        return jsonify({'error': f'{error_msg}: {str(e)}'}), 500


@app.route('/predict_video', methods=['POST'])
def predict_video():
    """处理视频预测请求 / Handle video prediction requests"""
    lang = get_language_from_request()

    try:
        if 'video' not in request.files:
            error_msg = msg_manager.get_message('no_file_uploaded', lang)
            return jsonify({'error': error_msg}), 400

        file = request.files['video']
        if file.filename == '':
            error_msg = msg_manager.get_message('no_file_selected', lang)
            return jsonify({'error': error_msg}), 400

        if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # 从请求中获取参数 / Get parameters from request
                max_frames = int(request.form.get('max_frames', 12))
                aggregation_method = request.form.get('aggregation_method', 'mean_prob')

                # 提取和处理视频帧 / Extract and process video frames
                processing_msg = msg_manager.get_message('video_processing_start', lang)
                print(f"{processing_msg}: {filename}")

                processed_frames = extract_and_process_video_frames(
                    filepath, sam_predictor, max_frames, lang
                )

                if not processed_frames:
                    error_msg = msg_manager.get_message('no_valid_frames', lang)
                    return jsonify({'error': error_msg}), 400

                extracting_msg = msg_manager.get_message('extracting_frames', lang)
                frames_unit = msg_manager.get_message('frames_unit', lang)
                print(f"{extracting_msg} {len(processed_frames)} {frames_unit}")

                # 对每一帧进行预测 / Predict each frame
                frame_predictions = []
                frame_weights = []

                for i, frame in enumerate(processed_frames):
                    predicting_msg = msg_manager.get_message('predicting_frame', lang)
                    print(f"{predicting_msg} {i+1}/{len(processed_frames)}")

                    probs, weights = predict_single_frame(
                        frame, models, device, transform, ffe
                    )
                    frame_predictions.append(probs)
                    frame_weights.append(weights)

                # 聚合多帧预测结果 / Aggregate multi-frame predictions
                final_pred, final_prob = aggregate_frame_predictions(
                    frame_predictions, method=aggregation_method
                )

                # 计算平均模型权重 / Calculate average model weights
                avg_weights = np.mean(frame_weights, axis=0)

                # 准备结果 / Prepare results
                class_names_en = ['mild', 'moderate', 'severe']
                class_names_zh = [msg_manager.get_message(name, 'zh') for name in class_names_en]
                predicted_class = class_names_zh[final_pred] if lang == 'zh' else class_names_en[final_pred]

                # 准备详细的帧级别预测信息 / Prepare detailed frame-level prediction information
                frame_details = []
                for i, (probs, weights) in enumerate(zip(frame_predictions, frame_weights)):
                    frame_pred_idx = np.argmax(probs)
                    frame_pred_class = class_names_zh[frame_pred_idx] if lang == 'zh' else class_names_en[frame_pred_idx]

                    frame_detail = {
                        'frame': i + 1,
                        'predicted_class': frame_pred_class,
                        'probabilities': {
                            msg_manager.get_message('mild', lang): float(probs[0]),
                            msg_manager.get_message('moderate', lang): float(probs[1]),
                            msg_manager.get_message('severe', lang): float(probs[2])
                        },
                        'confidence': float(np.max(probs))
                    }

                    # 添加英文对照 / Add English reference
                    if lang == 'zh':
                        frame_detail['predicted_class_en'] = class_names_en[frame_pred_idx]
                        frame_detail['probabilities_en'] = {
                            'mild': float(probs[0]),
                            'moderate': float(probs[1]),
                            'severe': float(probs[2])
                        }

                    frame_details.append(frame_detail)

                # 准备模型权重信息 / Prepare model weights information
                model_names = ['ConvNext', 'Swin', 'EfficientNet', 'ViT', 'YellowFeatures']
                model_weights_dict = {name: float(weight) for name, weight in zip(model_names, avg_weights)}

                result = {
                    'predicted_class': predicted_class,
                    'probabilities': {
                        msg_manager.get_message('mild', lang): float(final_prob[0]),
                        msg_manager.get_message('moderate', lang): float(final_prob[1]),
                        msg_manager.get_message('severe', lang): float(final_prob[2])
                    },
                    'confidence': float(np.max(final_prob)),
                    'aggregation_method': aggregation_method,
                    'total_frames_processed': len(processed_frames),
                    'model_weights': model_weights_dict,
                    'frame_predictions': frame_details,
                    'language': lang
                }

                # 添加英文对照 / Add English reference
                if lang == 'zh':
                    result['predicted_class_en'] = class_names_en[final_pred]
                    result['probabilities_en'] = {
                        'mild': float(final_prob[0]),
                        'moderate': float(final_prob[1]),
                        'severe': float(final_prob[2])
                    }

                # 删除临时文件 / Delete temporary file
                os.remove(filepath)

                return jsonify(result)

            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                error_msg = msg_manager.get_message('video_processing_failed', lang)
                print(f"{error_msg}: {e}")
                traceback.print_exc()
                return jsonify({'error': f'{error_msg}: {str(e)}'}), 500

        error_msg = msg_manager.get_message('unsupported_format', lang)
        return jsonify({'error': error_msg}), 400

    except Exception as e:
        error_msg = msg_manager.get_message('request_failed', lang)
        print(f"{error_msg}: {e}")
        traceback.print_exc()
        return jsonify({'error': f'{error_msg}: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点 / Health check endpoint"""
    lang = get_language_from_request()

    device_status = str(device) if device else msg_manager.get_message('device_not_initialized', lang)

    result = {
        'status': 'healthy',
        'message': msg_manager.get_message('system_healthy', lang),
        'models_loaded': len(models) > 0,
        'sam_loaded': sam_predictor is not None,
        'device': device_status,
        'language': lang
    }

    # 添加双语信息 / Add bilingual information
    if lang == 'zh':
        result['message_en'] = msg_manager.get_message('system_healthy', 'en')
    else:
        result['message_zh'] = msg_manager.get_message('system_healthy', 'zh')

    return jsonify(result)


# ============ 主程序 / Main Program ============

if __name__ == '__main__':
    start_msg_zh = msg_manager.get_message('server_starting', 'zh')
    start_msg_en = msg_manager.get_message('server_starting', 'en')
    print(start_msg_zh)
    print(start_msg_en)

    try:
        # 初始化模型 / Initialize models
        initialize_models()

        # 启动Flask服务器 / Start Flask server
        success_msg_zh = msg_manager.get_message('server_success', 'zh')
        success_msg_en = msg_manager.get_message('server_success', 'en')
        print(f"\n{success_msg_zh}")
        print(f"{success_msg_en}")

        print("🌐 访问 http://localhost:5000 查看API状态")
        print("🌐 Visit http://localhost:5000 to check API status")
        print("📊 支持视频和图片分类 / Supports video and image classification")
        print("\nAPI端点 / API Endpoints:")
        print(f"  - POST /predict_image: {msg_manager.get_message('endpoint_image', 'zh')} / {msg_manager.get_message('endpoint_image', 'en')}")
        print(f"  - POST /predict_video: {msg_manager.get_message('endpoint_video', 'zh')} / {msg_manager.get_message('endpoint_video', 'en')}")
        print(f"  - GET /health: {msg_manager.get_message('endpoint_health', 'zh')} / {msg_manager.get_message('endpoint_health', 'en')}")
        print("\n语言参数 / Language Parameter:")
        print("  - 添加 language=zh 或 language=en 到请求中")
        print("  - Add language=zh or language=en to your requests")

        app.run(debug=False, host='0.0.0.0', port=5000)

    except Exception as e:
        failed_msg_zh = msg_manager.get_message('server_failed', 'zh')
        failed_msg_en = msg_manager.get_message('server_failed', 'en')
        print(f"\n{failed_msg_zh} / {failed_msg_en}: {e}")
        traceback.print_exc()