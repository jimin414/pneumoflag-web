import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CAM_SIZE = 160

cam_tfms = transforms.Compose([
    transforms.Resize((CAM_SIZE, CAM_SIZE)),
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

def gradcam_on_tensor(model, x, target_layer_name="layer4.1.conv2"):
    model.eval()

    # model의 실제 device에 맞춤
    model_device = next(model.parameters()).device
    x = x.to(model_device)

    activations, gradients = [], []

    def fwd_hook(m, i, o):
        activations.append(o)

    def bwd_hook(m, gi, go):
        gradients.append(go[0])

    target_layer = get_module_by_name(model, target_layer_name)
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(x)
    logit = logits[0, 0]

    model.zero_grad()
    logit.backward()

    h1.remove()
    h2.remove()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations[0]).sum(dim=1).squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)

    h, w = cam.shape
    y_grid, x_grid = np.ogrid[-h/2:h/2, -w/2:w/2]
    center_mask = np.exp(-(x_grid*x_grid + y_grid*y_grid) / (2 * (h/4)**2))
    cam = cam * center_mask

    if cam.max() > 0:
        cam[cam < (cam.max() * 0.4)] = 0

    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-10)

    cam = cv2.resize(cam, (CAM_SIZE, CAM_SIZE))

    prob = torch.sigmoid(logit).item()
    pred = 1 if prob >= 0.5 else 0

    return cam, prob, pred

def make_gradcam_overlay(model, image_pil: Image.Image, target_layer_name="layer4.1.conv2", alpha=0.4):
    img_pil = image_pil.convert("RGB")
    x = cam_tfms(img_pil).unsqueeze(0)

    cam, prob, pred = gradcam_on_tensor(model, x, target_layer_name=target_layer_name)

    img_resized = np.array(img_pil.resize((CAM_SIZE, CAM_SIZE)))
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(img_bgr, 0.6, heatmap, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    overlay_pil = Image.fromarray(overlay_rgb)
    return overlay_pil, prob, pred