import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images
from skimage import io
from skimage.transform import resize
import cv2
from color_histogram_analyzer import ColorHistogramAnalyzer

# Set matplotlib to not use interactive mode
plt.ioff()

# Helper functions for non-blocking analysis
def non_blocking_analyze_image(analyzer, image_path, save_dir=None):
    """
    Analyze a single image without blocking for interactive plot display.
    
    Args:
        analyzer: The ColorHistogramAnalyzer instance
        image_path (str): Path to the image file
        save_dir (str, optional): Directory to save the output plot
    """
    # Load the image
    image = analyzer.load_image(image_path)
    
    # Compute histogram
    histograms = analyzer.compute_histogram(image)
    
    # Prepare title and save path
    image_name = os.path.basename(image_path).split('.')[0]
    title = f"Color Histogram - {image_name} ({analyzer.color_space})"
    
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{image_name}_histogram_{analyzer.color_space}.png")
    
    # Plot histogram but don't display it - just save it
    if save_path:
        # Create figure
        channel_names = analyzer.channel_names[analyzer.color_space]
        color_maps = analyzer.color_maps[analyzer.color_space]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)
        
        for i, (hist, name, cmap) in enumerate(zip(histograms, channel_names, color_maps)):
            # Create x-axis values
            bin_edges = np.linspace(0, 255, analyzer.bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Use a safer way to get colors from colormaps
            colors = plt.cm.get_cmap(cmap)(np.linspace(0.2, 1, analyzer.bins))
            
            # Plot histogram
            axes[i].bar(
                bin_centers, 
                hist.flatten(), 
                width=(255 / analyzer.bins) * 0.8,
                alpha=0.7, 
                color=colors
            )
            
            # Set labels
            axes[i].set_title(f"{name} Channel")
            axes[i].set_xlabel("Pixel Value")
            axes[i].set_ylabel("Normalized Frequency")
            axes[i].set_xlim(0, 255)
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved histogram to {save_path}")
    
    return image, histograms

def non_blocking_compare_images(analyzer, image_path1, image_path2, method='correlation', save_dir=None):
    """
    Compare two images without blocking for interactive plot display.
    
    Args:
        analyzer: The ColorHistogramAnalyzer instance
        image_path1 (str): Path to the first image file
        image_path2 (str): Path to the second image file
        method (str): Comparison method
        save_dir (str, optional): Directory to save the output plot
    """
    # Load images
    image1 = analyzer.load_image(image_path1)
    image2 = analyzer.load_image(image_path2)
    
    # Compute histograms
    hist1 = analyzer.compute_histogram(image1)
    hist2 = analyzer.compute_histogram(image2)
    
    # Compare histograms
    scores = analyzer.compare_histograms(hist1, hist2, method)
    
    # Calculate average similarity score
    avg_score = np.mean(scores)
    
    # Prepare save path
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        img1_name = os.path.basename(image_path1).split('.')[0]
        img2_name = os.path.basename(image_path2).split('.')[0]
        save_path = os.path.join(save_dir, f"comparison_{img1_name}_{img2_name}_{analyzer.color_space}.png")
    
    # Plot comparison but just save it without showing
    if save_path:
        # Use a simpler layout to avoid gridspec issues
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot images on the top row
        # Display images based on color space
        if analyzer.color_space == 'HSV' or analyzer.color_space == 'Lab':
            # Convert back to RGB for display
            if analyzer.color_space == 'HSV':
                display_img1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
                display_img2 = cv2.cvtColor(image2, cv2.COLOR_HSV2RGB)
            else:  # Lab
                display_img1 = cv2.cvtColor(image1, cv2.COLOR_Lab2RGB)
                display_img2 = cv2.cvtColor(image2, cv2.COLOR_Lab2RGB)
        else:  # RGB
            display_img1 = image1
            display_img2 = image2
        
        # Show images on top row
        axes[0, 0].imshow(display_img1)
        axes[0, 0].set_title("Image 1")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(display_img2)
        axes[0, 1].set_title("Image 2")
        axes[0, 1].axis('off')
        
        # Keep the top-right plot empty
        axes[0, 2].axis('off')
        
        # Plot histograms for each channel on bottom row
        channel_names = analyzer.channel_names[analyzer.color_space]
        color_maps = analyzer.color_maps[analyzer.color_space]
        
        for i, (h1, h2, name, cmap) in enumerate(zip(hist1, hist2, channel_names, color_maps)):
            # Create x-axis values
            bin_edges = np.linspace(0, 255, analyzer.bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot histograms using safe colormap access
            hist_color1 = plt.cm.get_cmap(cmap)(0.7)
            hist_color2 = plt.cm.get_cmap(cmap)(0.3)
            
            axes[1, i].bar(
                bin_centers - (255 / analyzer.bins) * 0.2, 
                h1.flatten(), 
                width=(255 / analyzer.bins) * 0.4,
                alpha=0.7, 
                color=hist_color1,
                label="Image 1"
            )
            
            axes[1, i].bar(
                bin_centers + (255 / analyzer.bins) * 0.2, 
                h2.flatten(), 
                width=(255 / analyzer.bins) * 0.4,
                alpha=0.7, 
                color=hist_color2,
                label="Image 2"
            )
            
            # Set labels
            axes[1, i].set_title(f"{name} Channel - Similarity: {scores[i]:.4f}")
            axes[1, i].set_xlabel("Pixel Value")
            axes[1, i].set_ylabel("Frequency")
            axes[1, i].set_xlim(0, 255)
            axes[1, i].grid(alpha=0.3)
            axes[1, i].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved comparison to {save_path}")
    
    print(f"Comparison method: {method}")
    print(f"Channel similarity scores: {scores}")
    print(f"Average similarity score: {avg_score:.4f}")
    
    return scores, avg_score

# Create demo functions
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

def classify_images(train_dir, test_dir, bins=32, color_space='RGB', method='correlation'):
    """Classify images based on color histograms."""
    # This is a simplified version for the demo
    print(f"Running classification with {color_space} color space using {method} method")
    print("(Classification would normally be performed here)")
    print(f"Training directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    
    # Just to keep the demo simple, we'll simulate the classification
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print("\nSimulated Classification Results:")
    for filename in test_files:
        if "day" in filename:
            predicted = "day"
            score = 0.85 + np.random.random() * 0.1
        else:
            predicted = "night"
            score = 0.83 + np.random.random() * 0.1
        
        print(f"{filename}: Classified as {predicted} (score: {score:.4f})")

def run_demo():
    """Run a comprehensive demo of the color histogram analyzer."""
    print("=== Color Histogram Analysis Demo ===")
    
    # Create demo directories
    os.makedirs("demo_output", exist_ok=True)
    
    # 1. Basic analysis of a sample image
    print("\n1. Basic Image Analysis")
    sample_images = create_demo_directory("demo_output/samples")
    analyzer = ColorHistogramAnalyzer(bins=32, color_space='RGB')
    # Use non-blocking version instead of original
    non_blocking_analyze_image(analyzer, sample_images[0], "demo_output/analysis")
    
    # 2. Compare different color spaces
    print("\n2. Comparing Color Spaces")
    for color_space in ['RGB', 'HSV', 'Lab']:
        print(f"Analyzing in {color_space} color space")
        analyzer = ColorHistogramAnalyzer(bins=32, color_space=color_space)
        # Use non-blocking version
        non_blocking_analyze_image(analyzer, sample_images[0], f"demo_output/color_spaces/{color_space}")
    
    # 3. Create and analyze color variations
    print("\n3. Color Variations Analysis")
    create_color_variations(sample_images[0], "demo_output/variations")
    # Compare original with color variations
    analyzer = ColorHistogramAnalyzer(bins=32, color_space='RGB')
    original = os.path.join("demo_output/variations", f"sample_1_original.jpg")
    for variation in ["dark", "bright", "red_boost", "blue_boost", "grayscale"]:
        var_path = os.path.join("demo_output/variations", f"sample_1_{variation}.jpg")
        print(f"\nComparing original to {variation}:")
        # Use non-blocking version
        non_blocking_compare_images(analyzer, original, var_path, 
                            method='correlation', save_dir="demo_output/variation_comparisons")
    
    # 4. Day/Night classification demo
    print("\n4. Day/Night Classification Demo")
    create_day_night_dataset("demo_output/day_night")
    classify_images("demo_output/day_night/train", "demo_output/day_night/test", 
                   bins=32, color_space='HSV', method='correlation')
    
    print("\nDemo completed! Results saved in 'demo_output' directory")

if __name__ == "__main__":
    run_demo()