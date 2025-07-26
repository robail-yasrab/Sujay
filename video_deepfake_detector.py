import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class VideoProjectiveGeometryDetector:
    """
    Video deepfake detector based on projective geometry principles
    Adapted from "Shadows Don't Lie and Lines Can't Bend!" (CVPR 2024)
    """
    
    def __init__(self, num_frames_to_analyze=10):
        self.num_frames = num_frames_to_analyze
        self.lsd = cv2.createLineSegmentDetector()
        
    def extract_random_frames(self, video_path, num_frames=None):
        """Extract random frames from video for analysis"""
        if num_frames is None:
            num_frames = self.num_frames
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
        
        if total_frames < num_frames:
            frame_indices = list(range(total_frames))
        else:
            # Select random frames distributed across the video
            frame_indices = sorted(random.sample(range(total_frames), num_frames))
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize frame for consistent processing
                frame = cv2.resize(frame, (640, 480))
                frames.append(frame)
        
        cap.release()
        return frames, frame_indices
    
    def analyze_perspective_field(self, image):
        """Analyze perspective field inconsistencies"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines = self.lsd.detect(gray)[0]
        
        if lines is None or len(lines) < 5:
            return 0.5  # Neutral score when insufficient lines
            
        # Find vanishing points
        vanishing_points = self._find_vanishing_points(lines, image.shape[:2])
        
        # Calculate perspective score based on vanishing point consistency
        if len(vanishing_points) == 0:
            return 0.8  # High inconsistency score
        
        # Score based on vanishing point distribution
        h, w = image.shape[:2]
        image_center = np.array([w/2, h/2])
        
        scores = []
        for vp in vanishing_points:
            # Distance from image center (normalized)
            distance = np.linalg.norm(vp - image_center) / np.linalg.norm([w, h])
            scores.append(min(distance * 0.5, 1.0))
        
        return np.mean(scores) if scores else 0.5
    
    def _find_vanishing_points(self, lines, image_shape):
        """Find vanishing points from detected lines"""
        intersections = []
        
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                intersection = self._line_intersection(lines[i][0], lines[j][0])
                if intersection is not None:
                    intersections.append(intersection)
        
        if len(intersections) < 3:
            return []
        
        # Simple clustering to find vanishing points
        intersections = np.array(intersections)
        h, w = image_shape
        threshold = min(h, w) * 0.1
        
        clusters = []
        for point in intersections:
            added_to_cluster = False
            for cluster in clusters:
                distances = np.linalg.norm(np.array(cluster) - point, axis=1)
                if np.min(distances) < threshold:
                    cluster.append(point)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([point])
        
        # Get cluster centers as vanishing points
        vanishing_points = []
        for cluster in clusters:
            if len(cluster) >= 3:
                center = np.mean(cluster, axis=0)
                vanishing_points.append(center)
        
        return vanishing_points
    
    def _line_intersection(self, line1, line2):
        """Calculate intersection of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
            
        px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
        py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
        
        # Check bounds
        if -2000 <= px <= 3000 and -2000 <= py <= 3000:
            return np.array([px, py])
        return None
    
    def analyze_shadow_consistency(self, image):
        """Analyze object-shadow relationship consistency"""
        # Detect shadows using color analysis
        shadow_mask = self._detect_shadows(image)
        
        # Detect objects using simple edge detection
        objects = self._detect_objects(image)
        
        if len(objects) == 0:
            return 0.5
        
        # Calculate shadow consistency scores
        inconsistency_scores = []
        h, w = image.shape[:2]
        
        for obj_x, obj_y, obj_w, obj_h in objects:
            # Define search area for shadows
            search_radius = max(obj_w, obj_h)
            search_x1 = max(0, obj_x - search_radius//2)
            search_y1 = max(0, obj_y - search_radius//2)
            search_x2 = min(w, obj_x + obj_w + search_radius//2)
            search_y2 = min(h, obj_y + obj_h + search_radius//2)
            
            # Analyze shadow density around object
            if search_y2 > search_y1 and search_x2 > search_x1:
                search_area = shadow_mask[search_y1:search_y2, search_x1:search_x2]
                shadow_density = np.sum(search_area) / search_area.size
                
                # Expected shadow density (this is a simplified heuristic)
                expected_density = 0.15
                density_diff = abs(shadow_density - expected_density)
                inconsistency_scores.append(min(density_diff * 3, 1.0))
        
        return np.mean(inconsistency_scores) if inconsistency_scores else 0.5
    
    def _detect_shadows(self, image):
        """Detect shadow regions in the image"""
        # Convert to HSV and LAB color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Shadow detection criteria
        low_value = hsv[:,:,2] < 100  # Low brightness in HSV
        low_luminance = lab[:,:,0] < 100  # Low luminance in LAB
        
        # Combine criteria
        shadow_mask = (low_value & low_luminance).astype(np.uint8)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        return shadow_mask
    
    def _detect_objects(self, image):
        """Simple object detection using contours"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:  # Filter by area
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 5.0:
                    objects.append((x, y, w, h))
        
        return objects
    
    def analyze_line_consistency(self, image):
        """Analyze line consistency (lines shouldn't bend)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lines = self.lsd.detect(gray)[0]
        
        if lines is None or len(lines) < 3:
            return 0.5
        
        # Check line straightness
        straightness_scores = []
        
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if line_length < 20:  # Skip very short lines
                continue
            
            # Sample points along the line
            num_samples = max(5, int(line_length / 10))
            t_values = np.linspace(0, 1, num_samples)
            
            # Calculate expected points on straight line
            expected_x = x1 + t_values * (x2 - x1)
            expected_y = y1 + t_values * (y2 - y1)
            
            # For simplicity, assume the line is straight and score based on length
            # In a full implementation, you would check actual pixel intensities
            # along the line to detect bending
            straightness_score = min(line_length / 100, 1.0)
            straightness_scores.append(1.0 - straightness_score * 0.1)
        
        return np.mean(straightness_scores) if straightness_scores else 0.5
    
    def analyze_frame(self, frame):
        """Analyze a single frame for geometric inconsistencies"""
        # Three main geometric analyses
        perspective_score = self.analyze_perspective_field(frame)
        shadow_score = self.analyze_shadow_consistency(frame)
        line_score = self.analyze_line_consistency(frame)
        
        return {
            'perspective_score': perspective_score,
            'shadow_score': shadow_score,
            'line_score': line_score,
            'combined_score': (perspective_score + shadow_score + line_score) / 3.0
        }
    
    def detect_deepfake_video(self, video_path, threshold=0.6):
        """Main function to detect if a video is deepfake"""
        try:
            print(f"\nAnalyzing video: {os.path.basename(video_path)}")
            
            # Extract random frames
            frames, frame_indices = self.extract_random_frames(video_path)
            
            if len(frames) == 0:
                return False, 0.0, {}
            
            # Analyze each frame
            frame_results = []
            all_scores = []
            
            for i, frame in enumerate(tqdm(frames, desc="Analyzing frames")):
                scores = self.analyze_frame(frame)
                frame_results.append({
                    'frame_index': frame_indices[i],
                    'scores': scores
                })
                all_scores.append(scores['combined_score'])
            
            # Aggregate results
            avg_score = np.mean(all_scores)
            max_score = np.max(all_scores)
            std_score = np.std(all_scores)
            
            # Decision logic: fake if average score > threshold OR max score significantly high
            is_fake = avg_score > threshold or max_score > (threshold + 0.15)
            
            results = {
                'is_fake': is_fake,
                'avg_confidence': avg_score,
                'max_confidence': max_score,
                'std_confidence': std_score,
                'frame_results': frame_results,
                'all_scores': all_scores,
                'threshold_used': threshold
            }
            
            return is_fake, avg_score, results
            
        except Exception as e:
            print(f"Error analyzing video: {str(e)}")
            return False, 0.0, {'error': str(e)}
    
    def visualize_frame_analysis(self, frame, scores, save_path=None):
        """Visualize analysis results for a frame"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original frame
        axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        
        # Detected shadows
        shadow_mask = self._detect_shadows(frame)
        axes[0, 1].imshow(shadow_mask, cmap='gray')
        axes[0, 1].set_title('Detected Shadows')
        axes[0, 1].axis('off')
        
        # Detected lines
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lines = self.lsd.detect(gray)[0]
        line_img = frame.copy()
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        axes[0, 2].imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Detected Lines')
        axes[0, 2].axis('off')
        
        # Score visualization
        score_names = ['Perspective', 'Shadow', 'Line', 'Combined']
        score_values = [
            scores['perspective_score'], 
            scores['shadow_score'], 
            scores['line_score'],
            scores['combined_score']
        ]
        colors = ['red' if s > 0.6 else 'orange' if s > 0.4 else 'green' for s in score_values]
        
        axes[1, 0].bar(score_names, score_values, color=colors)
        axes[1, 0].set_title('Inconsistency Scores')
        axes[1, 0].set_ylabel('Score (0=Real, 1=Fake)')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axhline(y=0.6, color='red', linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Objects detection visualization
        objects = self._detect_objects(frame)
        obj_img = frame.copy()
        for x, y, w, h in objects:
            cv2.rectangle(obj_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        axes[1, 1].imshow(cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Detected Objects ({len(objects)})')
        axes[1, 1].axis('off')
        
        # Summary text
        axes[1, 2].text(0.1, 0.8, f"Combined Score: {scores['combined_score']:.3f}", 
                       fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.6, f"Prediction: {'FAKE' if scores['combined_score'] > 0.6 else 'REAL'}", 
                       fontsize=12, fontweight='bold', 
                       color='red' if scores['combined_score'] > 0.6 else 'green',
                       transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.4, f"Perspective: {scores['perspective_score']:.3f}", 
                       fontsize=10, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.3, f"Shadow: {scores['shadow_score']:.3f}", 
                       fontsize=10, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.2, f"Line: {scores['line_score']:.3f}", 
                       fontsize=10, transform=axes[1, 2].transAxes)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show() 