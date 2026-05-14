import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def evaluate_and_plot(original_path, corrupted_path, restored_path, max_frames=300):
    cap_orig = cv2.VideoCapture(original_path)
    cap_corr = cv2.VideoCapture(corrupted_path)
    cap_rest = cv2.VideoCapture(restored_path)
    
    if not (cap_orig.isOpened() and cap_corr.isOpened() and cap_rest.isOpened()):
        print("Videolardan biri veya birkaci acilamadi!")
        return
        
    psnr_corr_list, ssim_corr_list = [], []
    psnr_rest_list, ssim_rest_list = [], []
    
    count = 0
    pbar = tqdm(total=max_frames, desc="Metrikler Hesaplaniyor ve Ciziliyor")
    
    while count < max_frames:
        ret1, f_orig = cap_orig.read()
        ret2, f_corr = cap_corr.read()
        ret3, f_rest = cap_rest.read()
        
        if not (ret1 and ret2 and ret3):
            break
            
        # PSNR Hesaplama
        psnr_corr_list.append(cv2.PSNR(f_orig, f_corr))
        psnr_rest_list.append(cv2.PSNR(f_orig, f_rest))
        
        # SSIM Hesaplama (Grayscale)
        g_orig = cv2.cvtColor(f_orig, cv2.COLOR_BGR2GRAY)
        g_corr = cv2.cvtColor(f_corr, cv2.COLOR_BGR2GRAY)
        g_rest = cv2.cvtColor(f_rest, cv2.COLOR_BGR2GRAY)
        
        ssim_corr_list.append(ssim(g_orig, g_corr, full=True)[0])
        ssim_rest_list.append(ssim(g_orig, g_rest, full=True)[0])
        
        count += 1
        pbar.update(1)
        
    pbar.close()
    cap_orig.release()
    cap_corr.release()
    cap_rest.release()
    
    # Ortalama Degerleri Konsola Yazdir
    avg_psnr_c = np.mean(psnr_corr_list)
    avg_psnr_r = np.mean(psnr_rest_list)
    avg_ssim_c = np.mean(ssim_corr_list)
    avg_ssim_r = np.mean(ssim_rest_list)
    
    print("\n=== ORTALAMA SONUCLAR ===")
    print(f"Bozuk Video -> PSNR: {avg_psnr_c:.2f} dB, SSIM: {avg_ssim_c:.4f}")
    print(f"Onarilmis Video -> PSNR: {avg_psnr_r:.2f} dB, SSIM: {avg_ssim_r:.4f}")
    print(f"KAZANC -> PSNR: +{avg_psnr_r - avg_psnr_c:.2f} dB, SSIM: {avg_ssim_r - avg_ssim_c:.4f}")

    # Grafik Cizimi (Matplotlib)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('QoS Optimizasyonu: Video Restorasyon Performansi (Kare Bazli)', fontsize=16, fontweight='bold')
    
    # PSNR Grafigi
    ax1.plot(psnr_corr_list, label="Bozulmus (Packet Loss)", color='red', alpha=0.7, linewidth=1.5)
    ax1.plot(psnr_rest_list, label="Onarilmis (AI Restored)", color='green', alpha=0.8, linewidth=2)
    ax1.set_title("PSNR Karsilastirmasi (Daha Yuksek = Daha Iyi)", fontsize=12)
    ax1.set_xlabel("Video Karesi (Frame Index)")
    ax1.set_ylabel("PSNR Degeri (dB)")
    ax1.legend(loc='lower right')
    
    # SSIM Grafigi
    ax2.plot(ssim_corr_list, label="Bozulmus (Packet Loss)", color='red', alpha=0.7, linewidth=1.5)
    ax2.plot(ssim_rest_list, label="Onarilmis (AI Restored)", color='green', alpha=0.8, linewidth=2)
    ax2.set_title("SSIM Karsilastirmasi (1.0'a Yakin = Daha Iyi)", fontsize=12)
    ax2.set_xlabel("Video Karesi (Frame Index)")
    ax2.set_ylabel("SSIM Degeri")
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Cikti Klasorune PNG Olarak Kaydet
    output_png = os.path.join(os.path.dirname(restored_path), "qos_metrics_plot.png")
    plt.savefig(output_png, dpi=300)
    print(f"\n[+] Gorsel basariyla olusturuldu ve kaydedildi: {output_png}")

if __name__ == "__main__":
    original_video = r"C:\Users\Acer\Downloads\QOS\YTDown_YouTube_Rain-Walk-in-Amsterdam-4K-ASMR_Media_yIzn6Q-eku8_002_720p.mp4"
    corrupted_video = r"C:\Users\Acer\Downloads\QOS\output\corrupted_amsterdam_30sn.mp4"
    restored_video = r"C:\Users\Acer\Downloads\QOS\output\restored_amsterdam_30sn.mp4"
    
    evaluate_and_plot(original_video, corrupted_video, restored_video, max_frames=300)
