import os
import shutil
import cv2
import glob
import random
from sklearn.model_selection import train_test_split

# Configuration
INPUT_DATASET = r"c:\Users\HP\Desktop\skincare\psoriasis_balanced_dataset"
OUTPUT_DATASET = r"c:\Users\HP\Desktop\skincare\psoriasis_cnn_dataset"
CLASSES = ["mild", "moderate", "severe"]
IMG_SIZE = (224, 224)
SPLIT_RATIO = 0.8 # 80% Train

def process_and_save(image_paths, subset_name, class_name):
    """
    Reads, resizes, and saves images to the target folder with sequential naming.
    """
    target_dir = os.path.join(OUTPUT_DATASET, subset_name, class_name)
    os.makedirs(target_dir, exist_ok=True)
    
    count = 0
    for i, img_path in enumerate(image_paths):
        try:
            # Read
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Resize
            img = cv2.resize(img, IMG_SIZE)
            
            # Convert to RGB (OpenCV is BGR by default, but we'll save as standard format. 
            # If we wanted to check, we leave as BGR for cv2.imwrite which expects BGR).
            # The prompt asks to "Convert images to RGB format". 
            # IMPORTANT: cv2.imwrite EXPECTS BGR. 
            # If we convert to RGB (cv2.cvtColor(..., BGR2RGB)) and use imwrite, colors will be swapped.
            # So, we ensure the process logic is consistent. We just Resizing. 
            # The 'Format' on disk is standard (JPG/PNG).
            
            # Rename
            filename = f"{class_name}_{i+1:03d}.jpg"
            save_path = os.path.join(target_dir, filename)
            
            # Save (Default BGR input -> Correct Colors in .jpg)
            cv2.imwrite(save_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    return count

def main():
    if os.path.exists(OUTPUT_DATASET):
        print(f"Cleaning existing output: {OUTPUT_DATASET}")
        shutil.rmtree(OUTPUT_DATASET)
        
    print(f"Processing dataset from: {INPUT_DATASET}")
    
    total_train = 0
    total_test = 0
    
    summary = {}
    
    for category in CLASSES:
        print(f"\nProcessing Class: {category}")
        source_dir = os.path.join(INPUT_DATASET, category)
        images = glob.glob(os.path.join(source_dir, "*.*"))
        
        if not images:
            print(f"  No images found for {category}!")
            continue
            
        print(f"  Found {len(images)} images.")
        
        # Split
        train_imgs, test_imgs = train_test_split(images, train_size=SPLIT_RATIO, random_state=42, shuffle=True)
        
        print(f"  Splitting -> Train: {len(train_imgs)}, Test: {len(test_imgs)}")
        
        # Process Train
        n_train = process_and_save(train_imgs, "train", category)
        # Process Test
        n_test = process_and_save(test_imgs, "test", category)
        
        total_train += n_train
        total_test += n_test
        summary[category] = {"train": n_train, "test": n_test}
        
    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print("="*40)
    print(f"{'Class':<15} | {'Train':<10} | {'Test':<10}")
    print("-" * 40)
    for cat in CLASSES:
        print(f"{cat:<15} | {summary[cat]['train']:<10} | {summary[cat]['test']:<10}")
    print("-" * 40)
    print(f"{'TOTAL':<15} | {total_train:<10} | {total_test:<10}")
    
    print(f"\nDataset saved to: {OUTPUT_DATASET}")

if __name__ == "__main__":
    main()
