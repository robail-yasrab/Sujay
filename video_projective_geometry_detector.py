import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PerspectiveFieldAnalyzer:
    """Analyzes perspective field inconsistencies in images"""
    
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector()
        
    def detect_vanishing_points(self, image):
        """Detect vanishing points using line segment analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines = self.lsd.detect(gray)[0]
        
        if lines is None or len(lines) < 5:
            return [], 0.0
            
        # Convert lines to slope-intercept form and find intersections
        vanishing_points = []
        intersections = []
        
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]
                
                intersection = self._line_intersection(line1, line2)
                if intersection is not None:
                    intersections.append(intersection)
        
        if len(intersections) == 0:
            return [], 0.0
            
        # Cluster intersections to find vanishing points
        intersections = np.array(intersections)
        
        # Simple clustering based on distance threshold
        threshold = min(image.shape[:2]) * 0.1
        clusters = []
        
        for point in intersections:
            added_to_cluster = False
            for cluster in clusters:
                distances = np.linalg.norm(cluster - point, axis=1)
                if np.min(distances) < threshold:
                    cluster.append(point)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([point])
        
        # Get vanishing points as cluster centers
        vanishing_points = []
        for cluster in clusters:
            if len(cluster) >= 3:  # Minimum points for a valid vanishing point
                cluster = np.array(cluster)
                center = np.mean(cluster, axis=0)
                vanishing_points.append(center)
        
        # Calculate perspective field score
        pf_score = self._calculate_perspective_score(vanishing_points, image.shape[:2])
        
        return vanishing_points, pf_score
    
    def _line_intersection(self, line1, line2):
        """Calculate intersection of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
            
        px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
        py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
        
        # Check if intersection is within reasonable bounds
        h, w = 1080, 1920  # Assume max reasonable image size
        if -w <= px <= 2*w and -h <= py <= 2*h:
            return np.array([px, py])
        return None
    
    def _calculate_perspective_score(self, vanishing_points, image_shape):
        """Calculate perspective field inconsistency score"""
        if len(vanishing_points) == 0:
            return 1.0  # High inconsistency score
        
        h, w = image_shape
        image_center = np.array([w/2, h/2])
        
        # Score based on vanishing point distribution and position
        scores = []
        for vp in vanishing_points:
            # Distance from image center (normalized)
            distance = np.linalg.norm(vp - image_center) / np.linalg.norm([w, h])
            scores.append(min(distance, 2.0))  # Cap at 2.0
        
        return np.mean(scores) if scores else 1.0

class ObjectShadowAnalyzer:
    """Analyzes object-shadow relationship inconsistencies"""
    
    def __init__(self):
        # Load YOLO for object detection
        self.net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg') if os.path.exists('yolov4.weights') else None
        self.classes = self._load_coco_names()
        
    def _load_coco_names(self):
        """Load COCO class names"""
        classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                  'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        return classes
    
    def analyze_shadows(self, image):
        """Analyze object-shadow relationships"""
        # Detect shadows using color space analysis
        shadows = self._detect_shadows(image)
        
        # Detect objects (simplified version without YOLO)
        objects = self._detect_objects_simple(image)
        
        # Calculate shadow consistency score
        consistency_score = self._calculate_shadow_consistency(objects, shadows, image)
        
        return consistency_score
    
    def _detect_shadows(self, image):
        """Detect shadow regions using color and intensity analysis"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Shadow detection using multiple criteria
        # Criteria 1: Low value in HSV
        low_value = hsv[:,:,2] < 80
        
        # Criteria 2: Low luminance in LAB
        low_luminance = lab[:,:,0] < 80
        
        # Criteria 3: Color distortion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(image)
        color_diff = np.abs(b.astype(float) - gray) + np.abs(g.astype(float) - gray) + np.abs(r.astype(float) - gray)
        low_color_diff = color_diff < 30
        
        # Combine criteria
        shadow_mask = low_value & low_luminance & low_color_diff
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        return shadow_mask
    
    def _detect_objects_simple(self, image):
        """Simple object detection using edge detection and contours"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                objects.append((x, y, w, h))
        
        return objects
    
    def _calculate_shadow_consistency(self, objects, shadows, image):
        """Calculate consistency between objects and their shadows"""
        if len(objects) == 0:
            return 0.5  # Neutral score when no objects detected
        
        h, w = image.shape[:2]
        inconsistency_scores = []
        
        for obj_x, obj_y, obj_w, obj_h in objects:
            obj_center = (obj_x + obj_w//2, obj_y + obj_h//2)
            obj_bottom = (obj_x + obj_w//2, obj_y + obj_h)
            
            # Look for shadows around the object
            search_radius = max(obj_w, obj_h)
            
            # Define search area around object
            search_x1 = max(0, obj_x - search_radius)
            search_y1 = max(0, obj_y - search_radius)
            search_x2 = min(w, obj_x + obj_w + search_radius)
            search_y2 = min(h, obj_y + obj_h + search_radius)
            
            search_area = shadows[search_y1:search_y2, search_x1:search_x2]
            shadow_density = np.sum(search_area) / (search_area.size + 1e-6)
            
            # Score based on expected shadow presence and direction
            expected_shadow_density = 0.1  # Expected shadow density around objects
            density_diff = abs(shadow_density - expected_shadow_density)
            inconsistency_scores.append(density_diff)
        
        return np.mean(inconsistency_scores) if inconsistency_scores else 0.5

class LineConsistencyAnalyzer:
    """Analyzes line consistency and geometric relationships"""
    
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector()
    
    def analyze_lines(self, image):
        """Analyze line consistency in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines = self.lsd.detect(gray)[0]
        
        if lines is None or len(lines) < 3:
            return 0.5  # Neutral score when insufficient lines
        
        # Analyze line properties
        line_scores = []
        
        # Check for parallel line consistency
        parallel_score = self._check_parallel_lines(lines)
        line_scores.append(parallel_score)
        
        # Check for perpendicular line consistency
        perpendicular_score = self._check_perpendicular_lines(lines)
        line_scores.append(perpendicular_score)
        
        # Check for line curvature (lines shouldn't bend)
        curvature_score = self._check_line_curvature(lines, gray)
        line_scores.append(curvature_score)
        
        return np.mean(line_scores)
    
    def _check_parallel_lines(self, lines):
        """Check consistency of parallel lines"""
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)
        
        angles = np.array(angles)
        
        # Group similar angles (parallel lines)
        angle_groups = []
        threshold = np.pi / 12  # 15 degrees
        
        for angle in angles:
            added_to_group = False
            for group in angle_groups:
                if any(abs(angle - g) < threshold or abs(angle - g - np.pi) < threshold for g in group):
                    group.append(angle)
                    added_to_group = True
                    break
            if not added_to_group:
                angle_groups.append([angle])
        
        # Calculate consistency score
        if len(angle_groups) == 0:
            return 0.5
        
        # Score based on how well parallel lines maintain consistent angles
        consistency_scores = []
        for group in angle_groups:
            if len(group) > 1:
                group_std = np.std(group)
                consistency_scores.append(1.0 - min(group_std / threshold, 1.0))
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _check_perpendicular_lines(self, lines):
        """Check consistency of perpendicular line relationships"""
        if len(lines) < 2:
            return 0.5
        
        perpendicular_scores = []
        
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]
                
                angle1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0])
                angle2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0])
                
                angle_diff = abs(angle1 - angle2)
                angle_diff = min(angle_diff, np.pi - angle_diff)
                
                # Check if lines are approximately perpendicular
                if abs(angle_diff - np.pi/2) < np.pi/12:  # Within 15 degrees of 90 degrees
                    perpendicular_scores.append(1.0 - abs(angle_diff - np.pi/2) / (np.pi/12))
        
        return np.mean(perpendicular_scores) if perpendicular_scores else 0.5
    
    def _check_line_curvature(self, lines, gray_image):
        """Check if detected lines are actually straight (not curved)"""
        curvature_scores = []
        
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            
            # Sample points along the line
            num_points = max(3, int(np.sqrt((x2-x1)**2 + (y2-y1)**2) / 10))
            
            if num_points < 3:
                continue
                
            x_points = np.linspace(x1, x2, num_points).astype(int)
            y_points = np.linspace(y1, y2, num_points).astype(int)
            
            # Check if points are within image bounds
            valid_points = []
            for x, y in zip(x_points, y_points):
                if 0 <= x < gray_image.shape[1] and 0 <= y < gray_image.shape[0]:
                    valid_points.append((x, y))
            
            if len(valid_points) < 3:
                continue
            
            # Calculate deviation from straight line
            valid_points = np.array(valid_points)
            if len(valid_points) > 2:
                # Fit a line and calculate deviation
                A = np.vstack([valid_points[:, 0], np.ones(len(valid_points))]).T
                try:
                    m, c = np.linalg.lstsq(A, valid_points[:, 1], rcond=None)[0]
                    expected_y = m * valid_points[:, 0] + c
                    deviation = np.mean(np.abs(valid_points[:, 1] - expected_y))
                    
                    # Normalize deviation and convert to score
                    max_deviation = 10.0  # pixels
                    curvature_score = 1.0 - min(deviation / max_deviation, 1.0)
                    curvature_scores.append(curvature_score)
                except:
                    pass
        
        return np.mean(curvature_scores) if curvature_scores else 0.5

class VideoProjectiveGeometryDetector:
    """Main video deepfake detector using projective geometry"""
    
    def __init__(self, num_frames_to_analyze=10):
        self.perspective_analyzer = PerspectiveFieldAnalyzer()
        self.shadow_analyzer = ObjectShadowAnalyzer()
        self.line_analyzer = LineConsistencyAnalyzer()
        self.num_frames = num_frames_to_analyze
        
    def extract_random_frames(self, video_path, num_frames=None):
        """Extract random frames from video"""
        if num_frames is None:
            num_frames = self.num_frames
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            # If video has fewer frames than requested, use all frames
            frame_indices = list(range(total_frames))
        else:
            # Select random frames
            frame_indices = sorted(random.sample(range(total_frames), num_frames))
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames, frame_indices
    
    def analyze_frame(self, frame):
        """Analyze a single frame for geometric inconsistencies"""
        # Analyze perspective field
        _, perspective_score = self.perspective_analyzer.detect_vanishing_points(frame)
        
        # Analyze object-shadow relationships
        shadow_score = self.shadow_analyzer.analyze_shadows(frame)
        
        # Analyze line consistency
        line_score = self.line_analyzer.analyze_lines(frame)
        
        return {
            'perspective_score': perspective_score,
            'shadow_score': shadow_score,
            'line_score': line_score
        }
    
    def detect_deepfake_video(self, video_path, threshold=0.6):
        """Detect if a video is a deepfake"""
        try:
            frames, frame_indices = self.extract_random_frames(video_path)
            
            if len(frames) == 0:
                return False, 0.0, []
            
            frame_results = []
            all_scores = []
            
            print(f"Analyzing {len(frames)} frames from video: {os.path.basename(video_path)}")
            
            for i, frame in enumerate(tqdm(frames, desc="Analyzing frames")):
                scores = self.analyze_frame(frame)
                frame_results.append({
                    'frame_index': frame_indices[i],
                    'scores': scores
                })
                
                # Combine scores (higher values indicate more inconsistency/fake)
                combined_score = (scores['perspective_score'] + 
                                scores['shadow_score'] + 
                                scores['line_score']) / 3.0
                all_scores.append(combined_score)
            
            # Aggregate results across all frames
            avg_score = np.mean(all_scores)
            max_score = np.max(all_scores)
            std_score = np.std(all_scores)
            
            # Make final decision (you can adjust this logic)
            is_fake = avg_score > threshold or max_score > (threshold + 0.2)
            
            results = {
                'is_fake': is_fake,
                'confidence': avg_score,
                'max_confidence': max_score,
                'std_confidence': std_score,
                'frame_results': frame_results,
                'all_scores': all_scores
            }
            
            return is_fake, avg_score, results
            
        except Exception as e:
            print(f"Error analyzing video {video_path}: {str(e)}")
            return False, 0.0, {}
    
    def batch_analyze_videos(self, video_paths, labels=None):
        """Analyze multiple videos"""
        results = []
        
        for i, video_path in enumerate(video_paths):
            print(f"\n--- Analyzing video {i+1}/{len(video_paths)} ---")
            is_fake, confidence, detailed_results = self.detect_deepfake_video(video_path)
            
            result = {
                'video_path': video_path,
                'predicted_fake': is_fake,
                'confidence': confidence,
                'detailed_results': detailed_results
            }
            
            if labels is not None:
                result['true_label'] = labels[i]
                result['correct'] = (is_fake == labels[i])
            
            results.append(result)
            
            print(f"Result: {'FAKE' if is_fake else 'REAL'} (confidence: {confidence:.3f})")
        
        return results

def visualize_analysis(frame, scores, save_path=None):
    """Visualize the analysis results on a frame"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original frame
    axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    # Scores visualization
    score_names = ['Perspective', 'Shadow', 'Line']
    score_values = [scores['perspective_score'], scores['shadow_score'], scores['line_score']]
    colors = ['red' if s > 0.6 else 'yellow' if s > 0.4 else 'green' for s in score_values]
    
    axes[0, 1].bar(score_names, score_values, color=colors)
    axes[0, 1].set_title('Geometric Inconsistency Scores')
    axes[0, 1].set_ylabel('Inconsistency Score')
    axes[0, 1].set_ylim(0, 1)
    
    # Add horizontal line for threshold
    axes[0, 1].axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Fake Threshold')
    axes[0, 1].legend()
    
    # Shadow detection visualization
    shadow_analyzer = ObjectShadowAnalyzer()
    shadow_mask = shadow_analyzer._detect_shadows(frame)
    axes[1, 0].imshow(shadow_mask, cmap='gray')
    axes[1, 0].set_title('Detected Shadows')
    axes[1, 0].axis('off')
    
    # Line detection visualization
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)[0]
    
    line_image = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    axes[1, 1].imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Detected Lines')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

# Example usage and testing
def main():
    """Main function to demonstrate the video deepfake detector"""
    
    # Initialize detector
    detector = VideoProjectiveGeometryDetector(num_frames_to_analyze=5)
    
    # Example: Analyze a single video
    video_path = "sample_video.mp4"  # Replace with your video path
    
    if os.path.exists(video_path):
        print("=== Single Video Analysis ===")
        is_fake, confidence, results = detector.detect_deepfake_video(video_path)
        
        print(f"\nFinal Result:")
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Prediction: {'FAKE' if is_fake else 'REAL'}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Max Confidence: {results['max_confidence']:.3f}")
        print(f"Score Std: {results['std_confidence']:.3f}")
        
        # Visualize analysis for first frame
        if results['frame_results']:
            frames, _ = detector.extract_random_frames(video_path, num_frames=1)
            if frames:
                first_frame_scores = results['frame_results'][0]['scores']
                visualize_analysis(frames[0], first_frame_scores)
    
    else:
        print(f"Video file {video_path} not found. Creating sample analysis...")
        
        # Create a sample analysis with dummy data
        print("\n=== Sample Analysis Results ===")
        sample_scores = {
            'perspective_score': 0.75,
            'shadow_score': 0.82,
            'line_score': 0.68
        }
        
        avg_score = np.mean(list(sample_scores.values()))
        is_fake = avg_score > 0.6
        
        print(f"Sample Analysis:")
        print(f"Perspective Score: {sample_scores['perspective_score']:.3f}")
        print(f"Shadow Score: {sample_scores['shadow_score']:.3f}")
        print(f"Line Score: {sample_scores['line_score']:.3f}")
        print(f"Average Score: {avg_score:.3f}")
        print(f"Prediction: {'FAKE' if is_fake else 'REAL'}")
    
    # Example: Batch analysis
    print("\n=== Batch Analysis Example ===")
    video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]  # Replace with actual paths
    labels = [True, False, True]  # True = fake, False = real
    
    # Filter existing files
    existing_files = [f for f in video_files if os.path.exists(f)]
    
    if existing_files:
        results = detector.batch_analyze_videos(existing_files)
        
        # Calculate accuracy if labels provided
        if len(existing_files) == len(labels):
            correct = sum(1 for r in results if r.get('correct', False))
            accuracy = correct / len(results)
            print(f"\nBatch Analysis Accuracy: {accuracy:.3f}")
    else:
        print("No video files found for batch analysis.")
        print("To use this detector:")
        print("1. Place video files in the same directory")
        print("2. Update the video_path variable")
        print("3. Run the script again")

if __name__ == "__main__":
    main() 