import cv2
import numpy as np
import sys

def process_illustration(input_path, output_path="source-prepped.png"):
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not read image.")
        return
        
    # 1. Grab the background color from the top-left pixel
    bg_color = img[0, 0]
    
    # 2. Create a mask of the background (with a small tolerance for compression artifacts)
    lower_bound = np.clip(bg_color.astype(int) - 15, 0, 255).astype(np.uint8)
    upper_bound = np.clip(bg_color.astype(int) + 15, 0, 255).astype(np.uint8)
    bg_mask = cv2.inRange(img, lower_bound, upper_bound)
    
    # 3. Create a white background
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    
    # 4. Composite the subject onto the white background
    subject_mask = cv2.bitwise_not(bg_mask)
    composited = np.where(subject_mask[:, :, None] == 255, img, white_bg)
    
    # 5. Convert to grayscale and darken the subject (so flames render in ASCII)
    gray = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)
    gray[subject_mask == 255] = (gray[subject_mask == 255] * 0.6).astype(np.uint8)
    
    cv2.imwrite(output_path, gray)
    print(f"Saved optimized illustration to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prep_photo.py <input_image>")
    else:
        process_illustration(sys.argv[1])