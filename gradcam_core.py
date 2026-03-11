import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grad-CAM용 전처리
cam_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_module_by_name(model, name: str):
    cur = model
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur

@torch.no_grad()
def denorm_for_show(x_tensor: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_tensor.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x_tensor.device)[:, None, None]
    x = x_tensor * std + mean
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).detach().cpu().numpy()
    return x

def gradcam_on_tensor(model, x, target_layer_name="layer4.1.conv2"):
    model.eval()
    x = x.to(device)

    activations, gradients = [], []

    def fwd_hook(m, i, o):
        activations.append(o)

    def bwd_hook(m, gi, go):
        gradients.append(go[0])

    target_layer = get_module_by_name(model, target_layer_name)
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x.requires_grad_(True)
    logits = model(x)
    logit = logits[0, 0]

    model.zero_grad()
    logit.backward(retain_graph=False)

    h1.remove()
    h2.remove()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations[0]).sum(dim=1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)

    # 중앙 마스크
    h, w = cam.shape
    y_grid, x_grid = np.ogrid[-h/2:h/2, -w/2:w/2]
    center_mask = np.exp(-(x_grid*x_grid + y_grid*y_grid) / (2 * (h/4)**2))
    cam = cam * center_mask

    # 노이즈 컷오프
    if cam.max() > 0:
        cam[cam < (cam.max() * 0.4)] = 0

    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-10)

    cam = cv2.resize(cam, (224, 224))

    prob = torch.sigmoid(logit).item()
    pred = 1 if prob >= 0.5 else 0

    return cam, prob, pred

def make_gradcam_overlay(model, image_pil: Image.Image, target_layer_name="layer4.1.conv2", alpha=0.4):
    """
    입력: PIL 이미지
    출력: overlay된 PIL 이미지, prob, pred
    """
    img_pil = image_pil.convert("RGB")
    x = cam_tfms(img_pil).unsqueeze(0).to(device)

    cam, prob, pred = gradcam_on_tensor(model, x, target_layer_name=target_layer_name)

    img_224 = np.array(img_pil.resize((224, 224)))
    img_bgr = cv2.cvtColor(img_224, cv2.COLOR_RGB2BGR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(img_bgr, 0.6, heatmap, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    overlay_pil = Image.fromarray(overlay_rgb)
    return overlay_pil, prob, pred