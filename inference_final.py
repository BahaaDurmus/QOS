import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    def __init__(self, n_feats):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )
    def forward(self, x): return x + self.conv(x)

class VSRModel(nn.Module):
    def __init__(self):
        super(VSRModel, self).__init__()
        self.conv_first = nn.Conv2d(45, 64, 3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(8)])
        self.conv_last = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b, t * c, h, w) 
        out = F.relu(self.conv_first(x))
        out = self.res_blocks(out)
        out = self.conv_last(out)
        return out

def perfect_restoration_v3(model, frames_tensor, device):
    model.eval()
    with torch.no_grad():
        input_batch = frames_tensor.unsqueeze(0).to(device)
        output = model(input_batch).squeeze(0).cpu()
        
    original_lq = frames_tensor[7].cpu() 
    
    # Eşikleme, Dilation ve Blur işlemleri
    mask = (original_lq.sum(dim=0, keepdim=True) < 0.15).float()
    
    kernel_dilate = torch.ones((1, 1, 15, 15))
    mask_dilated = F.conv2d(mask.unsqueeze(0), kernel_dilate, padding=7).squeeze(0)
    mask_dilated = (mask_dilated > 0).float()
    
    kernel_blur = torch.ones((1, 1, 7, 7)) / 49.0
    mask_soft = F.conv2d(mask_dilated.unsqueeze(0), kernel_blur, padding=3).squeeze(0)
    
    final_output = (mask_soft * output) + ((1 - mask_soft) * original_lq)
    return final_output.clamp(0, 1)

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Output directory
    os.makedirs(r'C:\Users\Acer\Downloads\QOS\output', exist_ok=True)

    # Initialize model
    model = VSRModel().to(device)

    # Load weights
    model_path = r"C:\Users\Acer\Downloads\QOS\vsr_model_sharp_v2_ep10.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except TypeError:
        # Fallback if weights_only is not supported in this PyTorch version
        model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model weights loaded successfully.")

    base_dir = r"C:\Users\Acer\Downloads\QOS\dataset_vimeo\corrupted"
    output_dir = r"C:\Users\Acer\Downloads\QOS\output"
    
    transform = T.Compose([
        T.Resize((256, 448)),
        T.ToTensor()
    ])

    subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    print(f"Found {len(subdirs)} folders to process.")

    for folder_name in subdirs:
        frames_dir = os.path.join(base_dir, folder_name)
        frames = []
        
        missing_file = False
        for i in range(1, 16):
            frame_path = os.path.join(frames_dir, f"im{i}.png")
            if not os.path.exists(frame_path):
                print(f"Skipping {folder_name}: Eksik dosya {frame_path}")
                missing_file = True
                break
            img = Image.open(frame_path).convert('RGB')
            img_t = transform(img)
            frames.append(img_t)
            
        if missing_file:
            continue

        frames_tensor = torch.stack(frames)
        final_output = perfect_restoration_v3(model, frames_tensor, device)
        
        output_path = os.path.join(output_dir, f"final_result_{folder_name}.png")
        vutils.save_image(final_output, output_path)
        print(f"Processed {folder_name}")

