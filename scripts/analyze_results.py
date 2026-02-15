import cv2
import numpy as np
from pathlib import Path
import os

def analyze_images(directory):
    path = Path(directory)
    if not path.exists():
        print(f"Directory not found: {directory}")
        return

    print(f"Analyzing images in: {directory}")
    print(f"{'Filename':<25} | {'Resolution':<15} | {'Sat (Avg)':<10} | {'Blur (Var)':<10} | {'Status':<10}")
    print("-" * 80)

    files = sorted(list(path.glob("*.png")))
    
    issues = []

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"{f.name:<25} | {'ERROR':<15} | {'-':<10} | {'-':<10} | {'LOAD FAIL'}")
            issues.append(f"{f.name}: Failed to load")
            continue

        h, w = img.shape[:2]
        res = f"{w}x{h}"
        
        # Saturation Check (is it colored?)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].mean()
        
        # Blur Check (Laplacian Variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        status = "OK"
        if sat < 20:
            status = "LOW SAT"
            issues.append(f"{f.name}: Low saturation ({sat:.2f}) - possibly grayscale")
        if blur_var < 100: # Threshold depends on image type, but <100 is often blurry
            status = "BLURRY?"
            issues.append(f"{f.name}: Low detail/blur ({blur_var:.2f})")

        print(f"{f.name:<25} | {res:<15} | {sat:<10.2f} | {blur_var:<10.2f} | {status}")

    print("-" * 80)
    if issues:
        print("\nPossible Issues Found:")
        for i in issues:
            print(f"- {i}")
    else:
        print("\nNo obvious technical issues found (resolution, saturation, blur check passed).")

if __name__ == "__main__":
    # Target the specific chapter folder we just verified
    target_dir = r"c:\Users\Nyx\Desktop\MANGACOLOR\output\chapters\ch_2e2e5f9ee771"
    analyze_images(target_dir)
