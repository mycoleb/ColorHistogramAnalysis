import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images
from skimage import io
from skimage.transform import resize
from color_histogram_analyzer import ColorHistogramAnalyzer

def run_demo():
    """Run a comprehensive demo of the color histogram analyzer."""
    print("=== Color Histogram Analysis Demo ===")
    
    # Create demo directories
    os.makedirs("demo_output", exist_ok=True)
    
    # 1. Basic analysis of a sample image
    print("\n1. Basic Image Analysis")
    sample_images = create_demo_directory("demo_output/samples")
    analyzer = ColorHistogramAnalyzer(bins=32, color_space='RGB')
    analyzer.analyze_image(sample_images[0], "demo_output/analysis")
    
    # 2. Compare different color spaces
    print("\n2. Comparing Color Spaces")
    for color_space in ['RGB', 'HSV', 'Lab']:
        print(f"Analyzing in {color_space} color space")
        analyzer = ColorHistogramAnalyzer(bins=32, color_space=color_space)
        analyzer.analyze_image(sample_images[0], f"demo_output/color_spaces/{color_space}")
    
    # 3. Create and analyze color variations
    print("\n3. Color Variations Analysis")
    create_color_variations(sample_images[0], "demo_output/variations")
    # Compare original with color variations
    analyzer = ColorHistogramAnalyzer(bins=32, color_space='RGB')
    original = os.path.join("demo_output/variations", f"sample_1_original.jpg")
    for variation in ["dark", "bright", "red_boost", "blue_boost", "grayscale"]:
        var_path = os.path.join("demo_output/variations", f"sample_1_{variation}.jpg")
        print(f"\nComparing original to {variation}:")
        analyzer.compare_images(original, var_path, method='correlation', 
                              save_dir="demo_output/variation_comparisons")
    
    # 4. Day/Night classification demo
    print("\n4. Day/Night Classification Demo")
    create_day_night_dataset("demo_output/day_night")
    classify_images("demo_output/day_night/train", "demo_output/day_night/test", 
                   bins=32, color_space='HSV', method='correlation')
    
    print("\nDemo completed! Results saved in 'demo_output' directory")

if __name__ == "__main__":
    run_demo()

def create_demo_directory(output_dir):
    """Create a demo directory with sample images."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sample images from scikit-learn
    dataset = load_sample_images()
    images = dataset.images
    
    # Save images to output directory
    for i, image in enumerate(images):
        plt.imsave(os.path.join(output_dir, f"sample_{i+1}.jpg"), image)
    
    print(f"Created demo directory with {len(images)} sample images at {output_dir}")
    
    return [os.path.join(output_dir, f"sample_{i+1}.jpg") for i in range(len(images))]

def create_color_variations(image_path, output_dir):
    """Create variations of an image with different color characteristics."""
    # Load the image
    image = io.imread(image_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original
    image_name = os.path.basename(image_path).split('.')[0]
    io.imsave(os.path.join(output_dir, f"{image_name}_original.jpg"), image)
    
    # Create a darker version
    dark = np.clip(image * 0.6, 0, 255).astype(np.uint8)
    io.imsave(os.path.join(output_dir, f"{image_name}_dark.jpg"), dark)
    
    # Create a brighter version
    bright = np.clip(image * 1.5, 0, 255).astype(np.uint8)
    io.imsave(os.path.join(output_dir, f"{image_name}_bright.jpg"), bright)
    
    # Boost red channel
    red_boost = image.copy()
    red_boost[:, :, 0] = np.clip(red_boost[:, :, 0] * 1.5, 0, 255).astype(np.uint8)
    io.imsave(os.path.join(output_dir, f"{image_name}_red_boost.jpg"), red_boost)
    
    # Boost blue channel
    blue_boost = image.copy()
    blue_boost[:, :, 2] = np.clip(blue_boost[:, :, 2] * 1.5, 0, 255).astype(np.uint8)
    io.imsave(os.path.join(output_dir, f"{image_name}_blue_boost.jpg"), blue_boost)
    
    # Create a grayscale version (but keep as RGB)
    gray = np.mean(image, axis=2, keepdims=True).astype(np.uint8)
    gray = np.repeat(gray, 3, axis=2)
    io.imsave(os.path.join(output_dir, f"{image_name}_grayscale.jpg"), gray)
    
    print(f"Created 6 color variations in {output_dir}")

def create_day_night_dataset(output_dir):
    """Create a simple day/night image classification dataset."""
    # Create directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(os.path.join(train_dir, "day"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "night"), exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"Creating day/night classification dataset in {output_dir}")
    
    # Generate synthetic day images (bright, blue sky)
    for i in range(5):
        # Create day image (bright with blue sky)
        img = np.zeros((100, 150, 3), dtype=np.uint8)
        # Sky (blue)
        img[:50, :, 0] = np.random.randint(100, 150)  # R
        img[:50, :, 1] = np.random.randint(150, 200)  # G
        img[:50, :, 2] = np.random.randint(200, 255)  # B
        # Ground (green/brown)
        img[50:, :, 0] = np.random.randint(100, 150)  # R
        img[50:, :, 1] = np.random.randint(150, 200)  # G
        img[50:, :, 2] = np.random.randint(50, 100)   # B
        # Add sun
        center_x, center_y = np.random.randint(30, 120), np.random.randint(10, 30)
        radius = np.random.randint(10, 15)
        y, x = np.ogrid[:100, :150]
        mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)
        img[mask] = [255, 255, 200]  # Yellow sun
        
        # Save image
        io.imsave(os.path.join(train_dir, "day", f"day_{i+1}.jpg"), img)
    
    # Generate synthetic night images (dark, starry)
    for i in range(5):
        # Create night image (dark with stars)
        img = np.zeros((100, 150, 3), dtype=np.uint8)
        # Sky (dark blue)
        img[:, :, 0] = np.random.randint(0, 30)  # R
        img[:, :, 1] = np.random.randint(0, 30)  # G
        img[:, :, 2] = np.random.randint(30, 80)  # B
        
        # Add some stars (small white dots)
        for _ in range(30):
            star_x, star_y = np.random.randint(0, 150), np.random.randint(0, 60)
            img[star_y, star_x] = [255, 255, 255]  # White star
            
        # Add moon
        center_x, center_y = np.random.randint(30, 120), np.random.randint(10, 30)
        radius = np.random.randint(8, 12)
        y, x = np.ogrid[:100, :150]
        mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)
        img[mask] = [200, 200, 180]  # Pale yellow moon
        
        # Ground (very dark)
        img[60:, :, 0] = np.random.randint(10, 40)  # R
        img[60:, :, 1] = np.random.randint(10, 40)  # G
        img[60:, :, 2] = np.random.randint(10, 40)  # B
        
        # Save image
        io.imsave(os.path.join(train_dir, "night", f"night_{i+1}.jpg"), img)
    
    # Create test images (a mix of day and night)
    for i in range(3):
        # Day test image
        img = np.zeros((100, 150, 3), dtype=np.uint8)
        # Sky (blue)
        img[:50, :, 0] = np.random.randint(100, 150)  # R
        img[:50, :, 1] = np.random.randint(150, 200)  # G
        img[:50, :, 2] = np.random.randint(200, 255)  # B
        # Ground
        img[50:, :, 0] = np.random.randint(100, 150)  # R
        img[50:, :, 1] = np.random.randint(150, 200)  # G
        img[50:, :, 2] = np.random.randint(50, 100)   # B
        # Sun
        center_x, center_y = np.random.randint(30, 120), np.random.randint(10, 30)
        radius = np.random.randint(10, 15)
        y, x = np.ogrid[:100, :150]
        mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)
        img[mask] = [255, 255, 200]  # Yellow sun
        
        io.imsave(os.path.join(test_dir, f"test_day_{i+1}.jpg"), img)
        
        # Night test image
        img = np.zeros((100, 150, 3), dtype=np.uint8)
        # Sky (dark blue)
        img[:, :, 0] = np.random.randint(0, 30)  # R
        img[:, :, 1] = np.random.randint(0, 30)  # G
        img[:, :, 2] = np.random.randint(30, 80)  # B
        
        # Add some stars
        for _ in range(30):
            star_x, star_y = np.random.randint(0, 150), np.random.randint(0, 60)
            img[star_y, star_x] = [255, 255, 255]  # White star
            
        # Moon
        center_x, center_y = np.random.randint(30, 120), np.random.randint(10, 30)
        radius = np.random.randint(8, 12)
        y, x = np.ogrid[:100, :150]
        mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2)
        img[mask] = [200, 200, 180]  # Pale yellow moon
        
        # Ground (very dark)
        img[60:, :, 0] = np.random.randint(10, 40)  # R
        img[60:, :, 1] = np.random.randint(10, 40)  # G
        img[60:, :, 2] = np.random.randint(10, 40)  # B
        
        io.imsave(os.path.join(test_dir, f"test_night_{i+1}.jpg"), img)
    
    print(f"Created day/night dataset with 10 training images and 6 test images")