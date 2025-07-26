# Video Projective Geometry Deepfake Detector

This project adapts the image-based projective geometry method from ["Shadows Don't Lie and Lines Can't Bend!" (CVPR 2024)](https://projective-geometry.github.io/) for video deepfake detection.

## 🎯 Key Features

- **Projective Geometry Analysis**: Detects geometric inconsistencies in deepfakes
- **Three-Component Detection**:
  - 🔍 **Perspective Field Analysis**: Vanishing point inconsistencies
  - 🌑 **Shadow Consistency**: Object-shadow relationship analysis  
  - 📐 **Line Analysis**: Detects unnaturally bent lines
- **Random Frame Sampling**: Analyzes multiple random frames from videos
- **Visual Analysis**: Comprehensive visualization of detection results

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements_video_detector.txt

# Or install individually:
pip install opencv-python numpy matplotlib tqdm scikit-learn
```

### 2. Basic Usage

```python
from video_deepfake_detector import VideoProjectiveGeometryDetector

# Initialize detector
detector = VideoProjectiveGeometryDetector(num_frames_to_analyze=10)

# Analyze a video
is_fake, confidence, results = detector.detect_deepfake_video("your_video.mp4")

print(f"Prediction: {'FAKE' if is_fake else 'REAL'}")
print(f"Confidence: {confidence:.3f}")
```

### 3. Run Complete Demo

```bash
python sample_usage.py
```

This will:
- Create sample videos with and without geometric inconsistencies
- Analyze both videos using projective geometry principles
- Show detailed visualizations and comparisons
- Generate analysis reports

## 📊 How It Works

The detector analyzes geometric properties that deepfake generators often get wrong:

### 1. Perspective Field Analysis
- Detects inconsistent vanishing points
- Analyzes line convergence patterns
- Identifies perspective distortions

### 2. Shadow Consistency Analysis  
- Examines object-shadow relationships
- Detects inconsistent shadow directions
- Analyzes shadow density patterns

### 3. Line Consistency Analysis
- Ensures detected lines remain straight
- Identifies unnaturally curved lines
- Checks geometric consistency

## 🎬 Sample Video Analysis

The detector extracts random frames from videos and analyzes each frame's geometric properties:

```python
# Extract and analyze random frames
frames, frame_indices = detector.extract_random_frames(video_path, num_frames=8)

for frame in frames:
    scores = detector.analyze_frame(frame)
    print(f"Perspective: {scores['perspective_score']:.3f}")
    print(f"Shadow: {scores['shadow_score']:.3f}")  
    print(f"Line: {scores['line_score']:.3f}")
    print(f"Combined: {scores['combined_score']:.3f}")
```

## 📈 Visualization Features

The detector provides comprehensive visualizations:

- **Frame Analysis**: Shows detected shadows, lines, and objects
- **Score Distribution**: Plots inconsistency scores across frames
- **Comparison Charts**: Compares real vs fake video analysis
- **Summary Statistics**: Detailed analysis breakdown

## 🔧 Configuration Options

```python
# Configure analysis parameters
detector = VideoProjectiveGeometryDetector(
    num_frames_to_analyze=15  # Number of random frames to analyze
)

# Adjust detection threshold
is_fake, confidence, results = detector.detect_deepfake_video(
    video_path, 
    threshold=0.6  # Fake if score > 0.6
)
```

## 📁 Project Structure

```
├── video_deepfake_detector.py    # Main detector class
├── sample_usage.py               # Complete demo and examples
├── requirements_video_detector.txt   # Required packages
└── README_video_detector.md      # This file
```

## 🎯 Expected Results

The detector creates sample videos to demonstrate its capabilities:

- **Real Video**: Consistent geometry, shadows, and perspective
- **Fake Video**: Geometric inconsistencies simulating deepfake artifacts

Typical results:
```
Real Video:    Score: 0.35 → REAL ✓
Fake Video:    Score: 0.73 → FAKE ✓
```

## 🔬 Technical Details

### Frame Selection Strategy
- Extracts random frames distributed across the video
- Analyzes 5-15 frames (configurable)
- Aggregates results using statistical measures

### Scoring System
- Each component returns 0.0-1.0 (0=Real, 1=Fake)
- Combined score averages all three components
- Decision threshold typically 0.5-0.6

### Geometric Analysis Methods
1. **Line Segment Detection**: OpenCV's Line Segment Detector
2. **Shadow Detection**: HSV/LAB color space analysis
3. **Object Detection**: Contour-based simple detection
4. **Vanishing Point Clustering**: Intersection-based clustering

## ⚠️ Limitations

- **Simple Object Detection**: Uses basic contour detection (can be enhanced with YOLO)
- **Synthetic Testing**: Demo uses artificially created test videos
- **Processing Speed**: Analyzes multiple frames (slower than single-frame methods)
- **Video Quality**: Works best with clear, well-lit videos

## 🔮 Potential Improvements

1. **Advanced Object Detection**: Integrate YOLO/Detectron2
2. **Temporal Analysis**: Add motion consistency checks
3. **Face-Specific Analysis**: Add facial landmark consistency
4. **Machine Learning**: Train classifiers on extracted features
5. **Real Dataset Testing**: Evaluate on actual deepfake datasets

## 📚 References

- Original Paper: [Shadows Don't Lie and Lines Can't Bend! Generative Models don't know Projective Geometry...for now](https://projective-geometry.github.io/)
- CVPR 2024 Paper by Sarkar et al.
- [Original Code Repository](https://github.com/hanlinm2/projective-geometry)

## 🤝 Contributing

Feel free to contribute by:
- Testing on real deepfake datasets
- Improving object detection methods
- Adding temporal consistency analysis
- Optimizing performance

## 📄 License

This project extends the concepts from the original CVPR 2024 paper for educational and research purposes. 