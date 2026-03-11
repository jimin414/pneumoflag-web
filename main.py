import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import numpy as np
import base64
from io import BytesIO
from gradcam_core import make_gradcam_overlay

# CPU로 고정
device = torch.device("cpu")

app = FastAPI()

# 1. 모델 구조 정의 (Colab에서 사용한 것과 동일해야 함)
def get_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1) # 바이너리 분류용
    return model

# 2. 모델 로드 (CPU 환경 기준)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
checkpoint = torch.load("checkpoint_best.pt", map_location="cpu")  # 일단 cpu로
state = checkpoint

if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]

model.load_state_dict(state, strict=False) # 가중치 입히기
device = torch.device("cpu")
model.to(device)
model.eval()

# 3. TTA용 변형 설정 (사용자님의 "정상 TTA" 방식 반영)
tta_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 이미지 읽기
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    
    # TTA 실행 (데모용:3번 반복 추론)(원래 8번)
    probs = []
    with torch.no_grad():
        for _ in range(3):
            img_tta = tta_transforms(image).unsqueeze(0).to(device)
            logits = model(img_tta)
            prob = torch.sigmoid(logits).item() # 0~1 사이 확률
            probs.append(prob)
    
    # 통계 계산 (평균 mu, 표준편차 sigma)
    mu = np.mean(probs)
    sigma = np.std(probs)

    # 4. Reject 전략 적용 (시그마 기준)
    REJECT_THRESHOLD = 0.1  # 실험 결과에 따라 조정 가능
    is_rejected = bool(sigma > REJECT_THRESHOLD)
    
    result = "Pneumonia" if mu > 0.5 else "Normal"
    
    return {
        "prediction": result,
        "mean_probability": round(mu, 4),
        "uncertainty_sigma": round(sigma, 4),
        "is_rejected": is_rejected,
        "message": "⚠️ 불확실성이 높아 재촬영을 권장합니다." if is_rejected else "분석 완료"
    }

@app.get("/")
def home():
    return {"status": "Pneumoflag API is running"}

## gradcam. base64 변환 함수 추가
def pil_to_base64(img_pil):
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

## /gradcam 엔드포인트 추가
@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    overlay_pil, prob, pred = make_gradcam_overlay(
        model,
        image,
        target_layer_name="layer4.1.conv2",
        alpha=0.4,
    )

    heatmap_b64 = pil_to_base64(overlay_pil)

    return {
        "prediction": "Pneumonia" if pred == 1 else "Normal",
        "probability": round(float(prob), 4),
        "heatmap_png_base64": heatmap_b64
    }