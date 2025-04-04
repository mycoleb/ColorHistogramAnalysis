import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse

class ColorHistogramAnalyzer:
    """
    A class for analyzing and comparing color histograms of images.
    """
    
    def __init__(self, bins=32, color_space='RGB'):
        """
        Initialize the color histogram analyzer.
        
        Args:
            bins (int): Number of bins to use for the histogram (default: 32)
            color_space (str): Color space to use. Options: 'RGB', 'HSV', 'Lab' (default: 'RGB')
        """
        self.bins = bins
        self.color_space = color_space
        self.color_maps = {
            'RGB': ('Blues', 'Greens', 'Reds'),
            'HSV': ('Purples', 'Greens', 'Reds'),
            'Lab': ('Blues', 'Greens_r', 'Reds')
        }
        self.channel_names = {
            'RGB': ('Blue', 'Green', 'Red'),
            'HSV': ('Hue', 'Saturation', 'Value'),
            'Lab': ('Lightness', 'a (Green-Red)', 'b (Blue-Yellow)')
        }
    
    def load_image(self, image_path):
        """
        Load an image from the specified path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: The loaded image
        """
        # Read the image using OpenCV (BGR format)
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert the color space if needed
        if self.color_space == 'RGB':
            # Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.color_space == 'HSV':
            # Convert from BGR to HSV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'Lab':
            # Convert from BGR to Lab
            image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        else:
            raise ValueError(f"Unsupported color space: {self.color_space}")
        
        return image
    
    def compute_histogram(self, image):
        """
        Compute the color histogram of an image.
        
        Args:
            image (numpy.ndarray): The input image
            
        Returns:
            tuple: The histograms for each channel
        """
        histograms = []
        
        # Compute histogram for each channel
        for i in range(3):
            hist = cv2.calcHist(
                [image], 
                [i], 
                None, 
                [self.bins], 
                [0, 256]
            )
            # Normalize the histogram
            hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            histograms.append(hist)
        
        return histograms
    
    def compare_histograms(self, hist1, hist2, method='correlation'):
        """
        Compare two histograms using the specified method.
        
        Args:
            hist1 (list): First histogram
            hist2 (list): Second histogram
            method (str): Comparison method. Options: 'correlation', 'chi-square', 
                         'intersection', 'bhattacharyya' (default: 'correlation')
            
        Returns:
            list: Similarity scores for each channel
        """
        # Map method name to OpenCV comparison method
        method_map = {
            'correlation': cv2.HISTCMP_CORREL,
            'chi-square': cv2.HISTCMP_CHISQR,
            'intersection': cv2.HISTCMP_INTERSECT,
            'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA
        }
        
        if method not in method_map:
            raise ValueError(f"Unsupported comparison method: {method}")
        
        cv_method = method_map[method]
        
        # Compare each channel
        scores = []
        for i in range(3):
            score = cv2.compareHist(hist1[i], hist2[i], cv_method)
            scores.append(score)
        
        return scores
    
    def plot_histogram(self, histograms, title="Color Histogram", save_path=None):
        """
        Plot the color histograms.
        
        Args:
            histograms (tuple): Histograms for each channel
            title (str): Title for the plot (default: "Color Histogram")
            save_path (str, optional): Path to save the plot (default: None)
        """
        channel_names = self.channel_names[self.color_space]
        color_maps = self.color_maps[self.color_space]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)
        
        for i, (hist, name, cmap) in enumerate(zip(histograms, channel_names, color_maps)):
            # Create x-axis values
            bin_edges = np.linspace(0, 255, self.bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot histogram
            axes[i].bar(
                bin_centers, 
                hist.flatten(), 
                width=(255 / self.bins) * 0.8,
                alpha=0.7, 
                color=plt.cm.get_cmap(cmap)(np.linspace(0.2, 1, self.bins))
            )
            
            # Set labels
            axes[i].set_title(f"{name} Channel")
            axes[i].set_xlabel("Pixel Value")
            axes[i].set_ylabel("Normalized Frequency")
            axes[i].set_xlim(0, 255)
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_comparison(self, image1, image2, hist1, hist2, scores, save_path=None):
        """
        Plot two images and their histogram comparison.
        
        Args:
            image1 (numpy.ndarray): First image
            image2 (numpy.ndarray): Second image
            hist1 (tuple): Histograms for the first image
            hist2 (tuple): Histograms for the second image
            scores (list): Similarity scores for each channel
            save_path (str, optional): Path to save the plot (default: None)
        """
        # Create figure and axes
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 6)
        
        # Plot images
        ax1 = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:])
        
        # Display images based on color space
        if self.color_space == 'HSV' or self.color_space == 'Lab':
            # Convert back to RGB for display
            if self.color_space == 'HSV':
                display_img1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
                display_img2 = cv2.cvtColor(image2, cv2.COLOR_HSV2RGB)
            else:  # Lab
                display_img1 = cv2.cvtColor(image1, cv2.COLOR_Lab2RGB)
                display_img2 = cv2.cvtColor(image2, cv2.COLOR_Lab2RGB)
        else:  # RGB
            display_img1 = image1
            display_img2 = image2
        
        ax1.imshow(display_img1)
        ax1.set_title("Image 1")
        ax1.axis('off')
        
        ax2.imshow(display_img2)
        ax2.set_title("Image 2")
        ax2.axis('off')
        
        # Plot histograms for each channel
        channel_names = self.channel_names[self.color_space]
        color_maps = self.color_maps[self.color_space]
        
        for i, (h1, h2, name, cmap) in enumerate(zip(hist1, hist2, channel_names, color_maps)):
            ax = fig.add_subplot(gs[i+1, :])
            
            # Create x-axis values
            bin_edges = np.linspace(0, 255, self.bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot histograms
            ax.bar(
                bin_centers - (255 / self.bins) * 0.2, 
                h1.flatten(), 
                width=(255 / self.bins) * 0.4,
                alpha=0.7, 
                color=plt.cm.get_cmap(cmap)(0.7),
                label="Image 1"
            )
            
            ax.bar(
                bin_centers + (255 / self.bins) * 0.2, 
                h2.flatten(), 
                width=(255 / self.bins) * 0.4,
                alpha=0.7, 
                color=plt.cm.get_cmap(cmap)(0.3),
                label="Image 2"
            )
            
            # Set labels
            ax.set_title(f"{name} Channel - Similarity Score: {scores[i]:.4f}")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Normalized Frequency")
            ax.set_xlim(0, 255)
            ax.grid(alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_image(self, image_path, save_dir=None):
        """
        Analyze a single image and plot its color histogram.
        
        Args:
            image_path (str): Path to the image file
            save_dir (str, optional): Directory to save the output plot (default: None)
        """
        # Load the image
        image = self.load_image(image_path)
        
        # Compute histogram
        histograms = self.compute_histogram(image)
        
        # Prepare title and save path
        image_name = Path(image_path).stem
        title = f"Color Histogram - {image_name} ({self.color_space})"
        
        save_path = None
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{image_name}_histogram_{self.color_space}.png"
        
        # Plot histogram
        self.plot_histogram(histograms, title, save_path)
        
        return image, histograms
    
    def compare_images(self, image_path1, image_path2, method='correlation', save_dir=None):
        """
        Compare two images based on their color histograms.
        
        Args:
            image_path1 (str): Path to the first image file
            image_path2 (str): Path to the second image file
            method (str): Comparison method (default: 'correlation')
            save_dir (str, optional): Directory to save the output plot (default: None)
        """
        # Load images
        image1 = self.load_image(image_path1)
        image2 = self.load_image(image_path2)
        
        # Compute histograms
        hist1 = self.compute_histogram(image1)
        hist2 = self.compute_histogram(image2)
        
        # Compare histograms
        scores = self.compare_histograms(hist1, hist2, method)
        
        # Calculate average similarity score
        avg_score = np.mean(scores)
        
        # Prepare save path
        save_path = None
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            img1_name = Path(image_path1).stem
            img2_name = Path(image_path2).stem
            save_path = save_dir / f"comparison_{img1_name}_{img2_name}_{self.color_space}.png"
        
        # Plot comparison
        self.plot_comparison(image1, image2, hist1, hist2, scores, save_path)
        
        print(f"Comparison method: {method}")
        print(f"Channel similarity scores: {scores}")
        print(f"Average similarity score: {avg_score:.4f}")
        
        return scores, avg_score

def analyze_directory(directory, bins=32, color_space='RGB', save_dir=None):
    """
    Analyze all images in a directory and generate histograms.
    
    Args:
        directory (str): Path to the directory containing images
        bins (int): Number of bins for histograms (default: 32)
        color_space (str): Color space to use (default: 'RGB')
        save_dir (str, optional): Directory to save output plots (default: None)
    """
    # Initialize analyzer
    analyzer = ColorHistogramAnalyzer(bins=bins, color_space=color_space)
    
    # Get all image files in the directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(image_extensions)
    ]
    
    if not image_paths:
        print(f"No images found in {directory}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Analyze each image
    for image_path in image_paths:
        try:
            print(f"Analyzing {os.path.basename(image_path)}...")
            analyzer.analyze_image(image_path, save_dir)
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")

def compare_directory_pair(dir1, dir2, bins=32, color_space='RGB', method='correlation', save_dir=None):
    """
    Compare corresponding images in two directories based on their color histograms.
    
    Args:
        dir1 (str): Path to the first directory
        dir2 (str): Path to the second directory
        bins (int): Number of bins for histograms (default: 32)
        color_space (str): Color space to use (default: 'RGB')
        method (str): Comparison method (default: 'correlation')
        save_dir (str, optional): Directory to save output plots (default: None)
    """
    # Initialize analyzer
    analyzer = ColorHistogramAnalyzer(bins=bins, color_space=color_space)
    
    # Get all image files in the directories
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Get image files in first directory
    image_files1 = [
        f for f in os.listdir(dir1)
        if f.lower().endswith(image_extensions)
    ]
    
    # Get image files in second directory
    image_files2 = [
        f for f in os.listdir(dir2)
        if f.lower().endswith(image_extensions)
    ]
    
    # Find common filenames
    common_files = set(image_files1).intersection(set(image_files2))
    
    if not common_files:
        print("No matching image filenames found in both directories")
        return
    
    print(f"Found {len(common_files)} matching images")
    
    # Compare each pair of images
    results = []
    for filename in common_files:
        try:
            print(f"Comparing {filename}...")
            image_path1 = os.path.join(dir1, filename)
            image_path2 = os.path.join(dir2, filename)
            
            scores, avg_score = analyzer.compare_images(
                image_path1, image_path2, method, save_dir
            )
            
            results.append({
                'filename': filename,
                'scores': scores,
                'avg_score': avg_score
            })
        except Exception as e:
            print(f"Error comparing {filename}: {e}")
    
    # Print summary
    print("\nComparison Summary:")
    for result in sorted(results, key=lambda x: x['avg_score'], reverse=True):
        print(f"{result['filename']}: Average similarity score: {result['avg_score']:.4f}")

def classify_images(train_dir, test_dir, bins=32, color_space='RGB', method='correlation'):
    """
    Simple classifier for images based on color histograms.
    
    Args:
        train_dir (str): Path to directory containing training images organized in class folders
        test_dir (str): Path to directory containing test images
        bins (int): Number of bins for histograms (default: 32)
        color_space (str): Color space to use (default: 'RGB')
        method (str): Comparison method (default: 'correlation')
    """
    # Initialize analyzer
    analyzer = ColorHistogramAnalyzer(bins=bins, color_space=color_space)
    
    # Get class directories
    class_dirs = [
        d for d in os.listdir(train_dir) 
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    
    if not class_dirs:
        print(f"No class directories found in {train_dir}")
        return
    
    print(f"Found {len(class_dirs)} classes: {', '.join(class_dirs)}")
    
    # Dictionary to store training histograms
    training_data = {}
    
    # Process training images
    for class_name in class_dirs:
        class_path = os.path.join(train_dir, class_name)
        print(f"Processing training class: {class_name}")
        
        # Get image files in class directory
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_files:
            print(f"No images found in class directory {class_path}")
            continue
        
        print(f"Found {len(image_files)} training images for class {class_name}")
        
        # Store histograms for each training image
        class_histograms = []
        for filename in image_files:
            try:
                image_path = os.path.join(class_path, filename)
                image = analyzer.load_image(image_path)
                hist = analyzer.compute_histogram(image)
                class_histograms.append((filename, hist))
            except Exception as e:
                print(f"Error processing training image {filename}: {e}")
        
        training_data[class_name] = class_histograms
    
    # Process test images
    print("\nClassifying test images...")
    
    # Get test image files
    test_files = [
        f for f in os.listdir(test_dir)
        if f.lower().endswith(image_extensions)
    ]
    
    if not test_files:
        print(f"No test images found in {test_dir}")
        return
    
    print(f"Found {len(test_files)} test images")
    
    # Classify each test image
    results = []
    for filename in test_files:
        try:
            print(f"Classifying {filename}...")
            image_path = os.path.join(test_dir, filename)
            image = analyzer.load_image(image_path)
            hist = analyzer.compute_histogram(image)
            
            # Find best match among all training images
            best_score = -float('inf') if method == 'correlation' or method == 'intersection' else float('inf')
            best_class = None
            best_train_file = None
            
            for class_name, class_histograms in training_data.items():
                for train_filename, train_hist in class_histograms:
                    scores = analyzer.compare_histograms(hist, train_hist, method)
                    avg_score = np.mean(scores)
                    
                    # Update best match based on comparison method
                    if method in ['correlation', 'intersection']:
                        # Higher is better
                        if avg_score > best_score:
                            best_score = avg_score
                            best_class = class_name
                            best_train_file = train_filename
                    else:
                        # Lower is better
                        if avg_score < best_score:
                            best_score = avg_score
                            best_class = class_name
                            best_train_file = train_filename
            
            results.append({
                'filename': filename,
                'predicted_class': best_class,
                'best_match': best_train_file,
                'score': best_score
            })
            
            print(f"  Classified as {best_class} (score: {best_score:.4f})")
        except Exception as e:
            print(f"Error classifying {filename}: {e}")
    
    # Print classification summary
    print("\nClassification Results:")
    for result in results:
        print(f"{result['filename']}: Classified as {result['predicted_class']} (score: {result['score']:.4f})")

def main():
    parser = argparse.ArgumentParser(description='Color Histogram Analysis Tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Parser for analyzing a single image
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single image')
    analyze_parser.add_argument('image_path', help='Path to the image file')
    analyze_parser.add_argument('--bins', type=int, default=32, help='Number of histogram bins')
    analyze_parser.add_argument('--color-space', choices=['RGB', 'HSV', 'Lab'], default='RGB',
                               help='Color space to use')
    analyze_parser.add_argument('--save-dir', help='Directory to save output')
    
    # Parser for comparing two images
    compare_parser = subparsers.add_parser('compare', help='Compare two images')
    compare_parser.add_argument('image_path1', help='Path to the first image file')
    compare_parser.add_argument('image_path2', help='Path to the second image file')
    compare_parser.add_argument('--bins', type=int, default=32, help='Number of histogram bins')
    compare_parser.add_argument('--color-space', choices=['RGB', 'HSV', 'Lab'], default='RGB',
                               help='Color space to use')
    compare_parser.add_argument('--method', choices=['correlation', 'chi-square', 'intersection', 'bhattacharyya'],
                              default='correlation', help='Comparison method')
    compare_parser.add_argument('--save-dir', help='Directory to save output')
    
    # Parser for analyzing a directory of images
    analyze_dir_parser = subparsers.add_parser('analyze-dir', help='Analyze all images in a directory')
    analyze_dir_parser.add_argument('directory', help='Path to the directory containing images')
    analyze_dir_parser.add_argument('--bins', type=int, default=32, help='Number of histogram bins')
    analyze_dir_parser.add_argument('--color-space', choices=['RGB', 'HSV', 'Lab'], default='RGB',
                                   help='Color space to use')
    analyze_dir_parser.add_argument('--save-dir', help='Directory to save output')
    
    # Parser for comparing images in two directories
    compare_dir_parser = subparsers.add_parser('compare-dirs', 
                                             help='Compare corresponding images in two directories')
    compare_dir_parser.add_argument('dir1', help='Path to the first directory')
    compare_dir_parser.add_argument('dir2', help='Path to the second directory')
    compare_dir_parser.add_argument('--bins', type=int, default=32, help='Number of histogram bins')
    compare_dir_parser.add_argument('--color-space', choices=['RGB', 'HSV', 'Lab'], default='RGB',
                                   help='Color space to use')
    compare_dir_parser.add_argument('--method', choices=['correlation', 'chi-square', 'intersection', 'bhattacharyya'],
                                  default='correlation', help='Comparison method')
    compare_dir_parser.add_argument('--save-dir', help='Directory to save output')
    
    # Parser for classifying images
    classify_parser = subparsers.add_parser('classify', 
                                          help='Classify images based on color histograms')
    classify_parser.add_argument('train_dir', 
                               help='Path to directory containing training images organized in class folders')
    classify_parser.add_argument('test_dir', help='Path to directory containing test images')
    classify_parser.add_argument('--bins', type=int, default=32, help='Number of histogram bins')
    classify_parser.add_argument('--color-space', choices=['RGB', 'HSV', 'Lab'], default='RGB',
                               help='Color space to use')
    classify_parser.add_argument('--method', choices=['correlation', 'chi-square', 'intersection', 'bhattacharyya'],
                              default='correlation', help='Comparison method')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyzer = ColorHistogramAnalyzer(bins=args.bins, color_space=args.color_space)
        analyzer.analyze_image(args.image_path, args.save_dir)
    
    elif args.command == 'compare':
        analyzer = ColorHistogramAnalyzer(bins=args.bins, color_space=args.color_space)
        analyzer.compare_images(args.image_path1, args.image_path2, args.method, args.save_dir)
    
    elif args.command == 'analyze-dir':
        analyze_directory(args.directory, args.bins, args.color_space, args.save_dir)
    
    elif args.command == 'compare-dirs':
        compare_directory_pair(args.dir1, args.dir2, args.bins, args.color_space, 
                              args.method, args.save_dir)
    
    elif args.command == 'classify':
        classify_images(args.train_dir, args.test_dir, args.bins, args.color_space, args.method)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()