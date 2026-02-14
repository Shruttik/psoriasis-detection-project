import os
import cv2
import numpy as np
import shutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import glob
from concurrent.futures import ThreadPoolExecutor

# Configuration
INPUT_FOLDER = r"c:\Users\HP\Desktop\skincare\Psoriasis"
OUTPUT_FOLDER = r"c:\Users\HP\Desktop\skincare\psoriasis_balanced_dataset"
TARGET_COUNT = 1000

# Ensure output format
CLASSES = ["mild", "moderate", "severe"]

def get_image_paths(folder):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True)) # Handle nested
    return list(set(files))

def extract_features(image_path):
    """
    Extracts dermatology-relevant features:
    1. Redness (Mean 'a' channel in region of interest)
    2. Scaling/Texture (Laplacian Variance)
    3. Lesion Area (Relative size of redness)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.resize(img, (224, 224))
        
        # Convert to LAB for redness analysis
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Simple redness segmentation (Otsu's on 'a' channel often captures red lesions)
        _, mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Feature 1: Redness Intensity (Mean of 'a' channel inside the mask)
        mean_redness = cv2.mean(a, mask=mask)[0]
        
        # Feature 2: Lesion Area (Pixel count of mask / Total pixels)
        total_pixels = img.shape[0] * img.shape[1]
        lesion_area_ratio = cv2.countNonZero(mask) / total_pixels
        
        # Feature 3: Scaling/Texture (Laplacian Variance of the grayscale image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return [mean_redness, lesion_area_ratio, texture_score]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def augment_image(img):
    """Applies random augmentation: H-Flip, Rotate, Zoom / Brightness"""
    rows, cols, _ = img.shape
    
    # Random operations
    ops = []
    
    # 1. Flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        
    # 2. Rotation (+/- 10 degrees)
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    
    # 3. Brightness
    alpha = np.random.uniform(0.9, 1.1) # Contrast control
    beta = np.random.uniform(-10, 10)   # Brightness control
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    return img

def main():
    print("Finding images...")
    image_paths = get_image_paths(INPUT_FOLDER)
    print(f"Found {len(image_paths)} images.")
    
    if len(image_paths) == 0:
        print("No images found! Exiting.")
        return

    # Extract features
    print("Extracting features (Redness, Area, Texture)...")
    data = []
    valid_paths = []
    
    # Using ThreadPool for speed
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(extract_features, image_paths))
        
    for path, feats in zip(image_paths, results):
        if feats is not None:
            data.append(feats)
            valid_paths.append(path)
            
    if not data:
        print("Feature extraction failed for all images.")
        return

    # Normalize data for K-Means
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # K-Means Clustering
    print("Clustering images into 3 severity levels...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    
    # Determine which cluster corresponds to which Label
    # We assume:
    # Severe = Highest Redness, Largest Area
    # Mild = Lowest Redness, Smallest Area
    # We'll compute the mean feature vector for each cluster index
    cluster_means = []
    for i in range(3):
        cluster_data = scaled_data[labels == i]
        mean_feats = np.mean(cluster_data, axis=0)
        # Sum of standardized features (Redness + Area + Texture) is a decent heuristic for severity
        severity_score = np.sum(mean_feats)
        cluster_means.append((i, severity_score))
        
    # Sort clusters by severity score
    cluster_means.sort(key=lambda x: x[1])
    
    # Map cluster index to label string
    # Low score -> Mild, Mid -> Moderate, High -> Severe
    label_map = {
        cluster_means[0][0]: "mild",
        cluster_means[1][0]: "moderate",
        cluster_means[2][0]: "severe"
    }
    
    # Organize into lists
    categorized_images = {"mild": [], "moderate": [], "severe": []}
    for i, path in enumerate(valid_paths):
        cluster_idx = labels[i]
        label_name = label_map[cluster_idx]
        categorized_images[label_name].append(path)
        
    print("Initial Classification Counts:")
    for k, v in categorized_images.items():
        print(f"  {k}: {len(v)}")
        
    # Create Output Directory
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)
    
    # Process Balancing and Saving
    print("\nBalancing and Saving Dataset...")
    
    for category in CLASSES:
        cat_dir = os.path.join(OUTPUT_FOLDER, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        current_images = categorized_images[category]
        count = len(current_images)
        
        # 1. Downsample if > TARGET_COUNT
        if count > TARGET_COUNT:
            # Random sample
            np.random.shuffle(current_images)
            selected_images = current_images[:int(TARGET_COUNT * 1.1)] # Keep slightly more approx 1100 as per tolerance
            # Limit to exactly 1100 if user said 1000-1100, let's aim for 1050 to be safe
            selected_images = selected_images[:1050]
        else:
            selected_images = current_images[:]
            
        # Write original images
        write_idx = 1
        saved_images = [] # keep track for augmentation base
        
        for src_path in selected_images:
            dst_name = f"{category}_{write_idx:03d}.jpg"
            dst_path = os.path.join(cat_dir, dst_name)
            
            # Read and save to normalize ext/quality
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved_images.append(img)
                write_idx += 1
            else:
                 # Just copy if read fails (shouldn't happen given earlier check)
                shutil.copy(src_path, dst_path)
                write_idx += 1
                
        # 2. Augment if < TARGET_COUNT
        current_saved_count = len(saved_images)
        needed = TARGET_COUNT - current_saved_count
        
        if needed > 0 and current_saved_count > 0:
            print(f"  Augmenting {category}: Needs {needed} more images...")
            aug_idx = 0
            while needed > 0:
                # Cycle through existing images to create augments
                base_img = saved_images[aug_idx % current_saved_count]
                aug_img = augment_image(base_img)
                
                dst_name = f"{category}_{write_idx:03d}.jpg"
                dst_path = os.path.join(cat_dir, dst_name)
                
                cv2.imwrite(dst_path, aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                write_idx += 1
                needed -= 1
                aug_idx += 1
                
    print("\nProcessing Complete.")
    print("Final Counts:")
    for category in CLASSES:
        n = len(glob.glob(os.path.join(OUTPUT_FOLDER, category, "*")))
        print(f"  {category}: {n}")

if __name__ == "__main__":
    main()
