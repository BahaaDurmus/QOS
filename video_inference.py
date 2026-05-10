import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
from collections import deque

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


class VideoRestorer:
    """
    Önceden eğitilmiş VSR modelini kullanarak .mp4 videolarını restore eden sınıf.
    Kayan pencere (Sliding Window) yöntemiyle 15'er karelik gruplar halinde işler.
    """
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VSRModel().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except TypeError:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"[+] Model başarıyla yüklendi. Cihaz: {self.device}")
        
        # Modele girecek olan boyutlandırma
        self.transform = T.Compose([
            T.Resize((256, 448)),
            T.ToTensor()
        ])
        
    def process_video(self, input_mp4, output_mp4):
        """
        input_mp4 yolundaki videoyu okur, her bir karesini modele sokar
        ve temizlenmiş yeni videoyu output_mp4 yoluna kaydeder.
        """
        if not os.path.exists(input_mp4):
            raise FileNotFoundError(f"Girdi video dosyası bulunamadı: {input_mp4}")
            
        cap = cv2.VideoCapture(input_mp4)
        if not cap.isOpened():
            raise ValueError(f"Video dosyası açılamadı: {input_mp4}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[*] Video açıldı: {total_frames} kare, {fps} FPS")
        
        # Tüm kareleri RAM'e almak yerine deque ile stream yapabiliriz 
        # ancak kolaylık ve hız açısından bu örnekte hepsini RAM'de tutuyoruz.
        # (Çok uzun videolarda bellek yetersizliği olabilir, kısa klipler için idealdir)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR -> RGB -> PIL -> Tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            tensor_img = self.transform(pil_img)
            frames.append(tensor_img)
            
        cap.release()
        
        if len(frames) == 0:
            print("Video okunamadı veya kare bulunamadı.")
            return
            
        out_w, out_h = 448, 256
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4 formatı
        
        os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
        out_video = cv2.VideoWriter(output_mp4, fourcc, fps, (out_w, out_h))
        
        # Baştan ve sondan 7 kareyi tekrarlayarak padding yapalım 
        # (Model 15 karelik pencerenin tam ortasındaki 8. kareyi üretir)
        pad_start = [frames[0]] * 7
        pad_end = [frames[-1]] * 7
        padded_frames = pad_start + frames + pad_end
        
        print(f"[*] İşlem başlıyor... Toplam {len(frames)} kare işlenecek.")
        
        for i in range(len(frames)):
            # 15 karelik kayan pencere
            window = padded_frames[i : i + 15]
            window_tensor = torch.stack(window)
            
            # Model inference
            restored_tensor = perfect_restoration_v3(self.model, window_tensor, self.device)
            
            # Çıktı tensörünü [0,1] float -> [0,255] uint8 numpy'a çevirme
            restored_np = restored_tensor.permute(1, 2, 0).cpu().numpy()
            restored_np = (restored_np * 255.0).astype(np.uint8)
            
            # RGB -> BGR yapıp OpenCV ile yazma
            restored_bgr = cv2.cvtColor(restored_np, cv2.COLOR_RGB2BGR)
            out_video.write(restored_bgr)
            
            if (i + 1) % 20 == 0 or (i + 1) == len(frames):
                print(f"    -> İşlenen: {i+1}/{len(frames)}")
                
        out_video.release()
        print(f"[+] İşlem tamamlandı! Temizlenmiş video kaydedildi: {output_mp4}")


if __name__ == "__main__":
    # ------ KULLANIM ÖRNEĞİ ------
    
    # 1. Model ağırlıklarını ve Restorer sınıfını başlatın
    model_weights_path = r"C:\Users\Acer\Downloads\QOS\vsr_model_sharp_v2_ep10.pth"
    restorer = VideoRestorer(model_weights_path)
    
    # 2. İşlenecek mp4 dosyasını ve çıktı yolunu belirleyin
    input_video_path = r"C:\Users\Acer\Downloads\QOS\bozuk_video_test.mp4"
    output_video_path = r"C:\Users\Acer\Downloads\QOS\output\temizlenmis_video.mp4"
    
    # (Aşağıdaki satırları gerçek bir MP4 videosu olduğunda çalıştırabilirsiniz)
    if os.path.exists(input_video_path):
        restorer.process_video(input_video_path, output_video_path)
    else:
        print(f"Uyarı: {input_video_path} dosyası bulunamadı.")
        print("Lütfen 'input_video_path' değişkenini test etmek istediğiniz MP4 yoluna göre güncelleyip kodu çalıştırın.")
