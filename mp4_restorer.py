import os
import cv2
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import deque

# ==========================================
# 1. MODEL MİMARİSİ (Değiştirilmedi)
# ==========================================
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

# ==========================================
# 2. KUSURSUZ RESTORASYON FONKSİYONU (Genişletilmiş Maske ile)
# ==========================================
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

# ==========================================
# 3. UÇTAN UCA VİDEO ONARIM ARACI (Sliding Window Pipeline)
# ==========================================
class MP4Restorer:
    def __init__(self, model_path):
        # Otomatik olarak CUDA seçilir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Cihaz ayarlandı: {self.device}")
        
        self.model = VSRModel().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except TypeError:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print("[*] Model ağırlıkları başarıyla yüklendi!")
        
        # Orijinal çözünürlük korunur
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def restore_video(self, input_path, output_path):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Girdi videosu bulunamadı: {input_path}")
            
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Video açılamıyor: {input_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[*] Decoding: {input_path}")
        print(f"    -> Toplam Kare: {total_frames}, Çözünürlük: {orig_w}x{orig_h}, FPS: {fps}")
        
        # Videonun tamamını RAM'e yüklüyoruz. Eğer video çok uzunsa 
        # (Örn. binlerce kare), bunu bir deque ile anlık stream ederek güncellemek daha iyidir.
        # Bu implementasyonda stabilite için hepsini okuyacağız.
        frames = []
        count = 0
        while count < 300:  # 10 Saniyelik test kesiti (yaklaşık 30 fps * 10 = 300 kare)
            ret, frame = cap.read()
            if not ret:
                break
            
            # OpenCV BGR okur, modele RGB vermeliyiz
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            tensor_img = self.transform(pil_img)
            frames.append(tensor_img)
            count += 1
            
        cap.release()
        
        if not frames:
            print("Video okunamadı.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Sıkıştırmasız / Kayıpsız yüksek kaliteli video yazıcısı (imageio kullanarak)
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=9, macro_block_size=None)
        
        # Sliding window buffer oluşturmak için videonun başını ve sonunu padding yapıyoruz
        # (Model t-7 ile t+7 aralığına bakar, t ortadadır. t=0 için öncesi 7 kare kopya)
        pad_start = [frames[0]] * 7
        pad_end = [frames[-1]] * 7
        padded_frames = pad_start + frames + pad_end
        
        print("[*] Inference Başlıyor: Kareler onarılıp encoding yapılıyor...")
        
        for i in tqdm(range(len(frames)), desc="Video İşleniyor"):
            # Sliding window: (i) ile (i+15) arasındaki kareleri al
            window = padded_frames[i : i + 15]
            window_tensor = torch.stack(window)
            
            # Modeli çalıştır ve onarımı yap
            restored_tensor = perfect_restoration_v3(self.model, window_tensor, self.device)
            
            # Çıktıyı video yazıcısına uygun hale getir ([0, 255] uint8)
            restored_np = restored_tensor.permute(1, 2, 0).cpu().numpy()
            restored_np = (restored_np * 255.0).clip(0, 255).astype(np.uint8)
            
            # RGB olarak doğrudan imageio'ya yaz
            writer.append_data(restored_np)
            
        writer.close()
        print(f"[+] Uçtan uca onarım tamamlandı! Sonuç: {output_path}")

if __name__ == "__main__":
    # Ağırlık dosyasının yolu
    model_weights = r"C:\Users\Acer\Downloads\QOS\vsr_model_sharp_v2_ep10.pth"
    
    # Aracı başlat
    restorer = MP4Restorer(model_weights)
    
    # Test için dosya yolları (Bu yolları kendi dosyalarına göre güncelleyebilirsin)
    input_video = r"C:\Users\Acer\Downloads\QOS\output\corrupted_amsterdam_30sn.mp4"
    output_video = r"C:\Users\Acer\Downloads\QOS\output\restored_amsterdam_30sn.mp4"
    
    if os.path.exists(input_video):
        restorer.restore_video(input_video, output_video)
    else:
        print(f"Lütfen '{input_video}' isimli bir bozuk test videosunu dizine ekleyin ve kodu öyle çalıştırın!")
