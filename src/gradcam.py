import sys; sys.path.insert(0, '/home/sahith/Projects/Hackathon'); import config
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class GradCAM:
    def __init__(self, model, target_layer_name='layer4'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        target_layer = None
        if hasattr(self.model, 'backbone'):
            for name, module in self.model.backbone.named_modules():
                if name == self.target_layer_name:
                    target_layer = module
                    break
        else:
            for name, module in self.model.named_modules():
                if name == self.target_layer_name:
                    target_layer = module
                    break
                    
        if target_layer is None:
            raise ValueError(f"Target layer {self.target_layer_name} not found in model.")
            
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            if output.shape[1] == config.NUM_ORDINAL_TASKS:
                probs = torch.sigmoid(output)
                from src.corn_loss import corn_probs_to_grade
                pred_grade, _ = corn_probs_to_grade(probs)
                target_class = pred_grade.item()
            else:
                target_class = output.argmax(dim=1).item()
                
        if output.shape[1] == config.NUM_ORDINAL_TASKS:
            if target_class == 0:
                target = -output[0, 0]
            elif target_class == config.NUM_ORDINAL_TASKS:
                target = output[0, target_class - 1]
            else:
                target = output[0, target_class - 1]
        else:
            target = output[0, target_class]
            
        target.backward()
        
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        cam = cam.squeeze().cpu().numpy()
        cam_min = np.min(cam)
        cam_max = np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

def overlay_heatmap(image, heatmap, alpha=0.5):
    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(heatmap)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_colored)
    
    image = image.convert('RGB')
    if image.size != heatmap_img.size:
        heatmap_img = heatmap_img.resize(image.size, Image.BILINEAR)
        
    overlay = Image.blend(image, heatmap_img, alpha=alpha)
    return overlay

def save_gradcam_visualization(image_path, model, transform, device, output_path, target_class=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    gradcam = GradCAM(model, config.GRADCAM_TARGET_LAYER)
    heatmap = gradcam.generate(input_tensor, target_class)
    
    overlay = overlay_heatmap(image, heatmap)
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
