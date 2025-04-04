# Color Histogram Analyzer

A Python tool for analyzing and comparing images based on their color distributions.

## Overview

This project provides tools for:

- Analyzing color histograms of images
- Comparing images based on their color distributions
- Classifying images using color histogram features
- Visualizing color histograms in different color spaces (RGB, HSV, Lab)

Color histograms are a simple yet powerful way to capture the color distribution in images and can be used for various computer vision tasks such as image categorization, content-based image retrieval, and scene classification.

## Features

- **Multiple color spaces**: Support for RGB, HSV, and Lab color spaces
- **Flexible binning**: Customizable histogram bin sizes for different levels of precision
- **Various comparison methods**: Correlation, Chi-square, Intersection, and Bhattacharyya distance
- **Batch processing**: Process entire directories of images
- **Classification**: Simple classification based on color histograms
- **Rich visualizations**: Detailed histogram plots with color-coded channels

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-image (for the demo script)
- scikit-learn (for the demo script)

Install the requirements:

```bash
pip install opencv-python numpy matplotlib scikit-image scikit-learn
```

## Usage

### Command-line Interface

The tool provides a command-line interface with several subcommands:

#### Analyze a single image

```bash
python color_histogram_analyzer.py analyze path/to/image.jpg --bins 32 --color-space RGB --save-dir output
```

#### Compare two images

```bash
python color_histogram_analyzer.py compare image1.jpg image2.jpg --bins 32 --color-space HSV --method correlation --save-dir output
```

#### Analyze all images in a directory

```bash
python color_histogram_analyzer.py analyze-dir path/to/images --bins 32 --color-space RGB --save-dir output
```

#### Compare corresponding images in two directories

```bash
python color_histogram_analyzer.py compare-dirs dir1 dir2 --bins 32 --color-space RGB --method correlation --save-dir output
```

#### Classify images based on color histograms

```bash
python color_histogram_analyzer.py classify train_dir test_dir --bins 32 --color-space HSV --method correlation
```

### Python API

You can also use the `ColorHistogramAnalyzer` class directly in your Python code:

```python
from color_histogram_analyzer import ColorHistogramAnalyzer

# Initialize the analyzer
analyzer = ColorHistogramAnalyzer(bins=32, color_space='RGB')

# Analyze a single image
image, histograms = analyzer.analyze_image('path/to/image.jpg', 'output')

# Compare two images
scores, avg_score = analyzer.compare_images('image1.jpg', 'image2.jpg', 
                                         method='correlation', 
                                         save_dir='output')
```

## Demo Script

The project includes a demo script that showcases various features:

```bash
python demo_script.py
```

The demo will:

1. Create sample images for analysis
2. Demonstrate histogram analysis in different color spaces
3. Create and analyze color variations of the same image
4. Build a simple day/night image classifier using color histograms

## Understanding Color Spaces

- **RGB**: The standard additive color model used in digital imaging
- **HSV**: Represents colors in terms of Hue, Saturation, and Value
- **Lab**: A color space designed to be perceptually uniform, where L represents lightness and a, b represent color dimensions

## Histogram Comparison Methods

- **Correlation**: Values between -1 and 1, where 1 means perfect match
- **Chi-Square**: Values between 0 and infinity, where 0 means perfect match
- **Intersection**: Values between 0 and 1 (for normalized histograms), where 1 means perfect match
- **Bhattacharyya distance**: Values between 0 and 1, where 0 means perfect match

## Applications

- Image classification based on color properties
- Finding similar images in a dataset
- Content-based image retrieval
- Scene recognition (day/night, indoor/outdoor)
- Image filtering and organization

## Project Structure

```
.
├── color_histogram_analyzer.py   # Main implementation
├── demo_script.py                # Demo script to showcase features
├── README.md                     # This file
└── demo_output/                  # Created by the demo script
    ├── analysis/                 # Basic image analysis results
    ├── color_spaces/             # Comparisons of different color spaces
    ├── samples/                  # Sample images
    ├── variations/               # Color variations of images
    ├── variation_comparisons/    # Comparisons between color variations
    └── day_night/                # Day/night classification demo
```

## License

This project is released under the MIT License.


## What skills are involved

This Color Histogram Analyzer project demonstrates:

1. **Computer Vision Fundamentals**: The project shows a solid understanding of color spaces (RGB, HSV, Lab) and how they represent images differently, which is fundamental to computer vision applications.

2. **Image Processing**: The code demonstrates techniques for loading, transforming, analyzing, and comparing images using established libraries like OpenCV.

3. **Histogram Analysis**: The project showcases histogram computation and comparison using various methods (correlation, chi-square, intersection, Bhattacharyya distance), demonstrating understanding of statistical image analysis.

4. **Data Visualization**: There's sophisticated use of matplotlib to create meaningful visualizations of image data and histogram comparisons with proper color-coding, labeling, and layout.

5. **Python Programming**: The code demonstrates strong Python skills including:
   - Object-oriented programming (class design)
   - Error handling
   - Command-line interface design
   - File I/O operations
   - Modular code organization

6. **Scientific Computing**: Proficient use of NumPy for numerical operations and array manipulations crucial for image processing.

7. **Machine Learning Applications**: The project implements a simple image classifier based on color features, demonstrating basic machine learning concepts without requiring deep learning frameworks.

8. **Command-Line Tool Design**: The implementation of a full-featured command-line tool with multiple subcommands and argument parsing shows software engineering skills.

9. **Documentation**: The code includes thorough docstrings and a comprehensive README, showing skills in technical writing and software documentation.

10. **Testing and Demo Creation**: The project includes a demo script to showcase functionality, demonstrating an understanding of how to validate and present software capabilities.

This project is relevant to computer vision, image processing, data visualization, or general software development where image analysis is involved.