#!/usr/bin/env python3
"""
Sample usage script for Video Projective Geometry Deepfake Detector
Based on "Shadows Don't Lie and Lines Can't Bend!" (CVPR 2024)
"""

import os
import cv2
import numpy as np
from video_deepfake_detector import VideoProjectiveGeometryDetector
import matplotlib.pyplot as plt

def create_sample_video(output_path, duration_seconds=5, fps=30):
    """Create a sample video for testing"""
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    print(f"Creating sample video: {output_path}")
    print(f"Duration: {duration_seconds}s, FPS: {fps}, Total frames: {total_frames}")
    
    for frame_num in range(total_frames):
        # Create a frame with geometric elements
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background gradient
        for y in range(height):
            intensity = int(50 + (y / height) * 100)
            frame[y, :] = [intensity, intensity + 20, intensity + 40]
        
        # Add some geometric shapes that will create lines and shadows
        time_factor = frame_num / total_frames
        
        # Moving rectangle (creates lines and shadows)
        rect_x = int(50 + time_factor * 400)
        rect_y = int(100 + 50 * np.sin(time_factor * 2 * np.pi))
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 100, rect_y + 80), 
                     (100, 150, 200), -1)
        
        # Add shadow for the rectangle
        shadow_offset = 20
        cv2.rectangle(frame, 
                     (rect_x + shadow_offset, rect_y + shadow_offset), 
                     (rect_x + 100 + shadow_offset, rect_y + 80 + shadow_offset), 
                     (30, 30, 30), -1)
        
        # Add some lines (for perspective analysis)
        # Horizontal lines
        for i in range(3):
            y_pos = 200 + i * 60
            cv2.line(frame, (0, y_pos), (width, y_pos), (255, 255, 255), 2)
        
        # Vertical lines
        for i in range(4):
            x_pos = 100 + i * 150
            cv2.line(frame, (x_pos, 0), (x_pos, height), (255, 255, 255), 2)
        
        # Add some diagonal lines for vanishing point
        vanish_x, vanish_y = int(width * 0.7), int(height * 0.3)
        for i in range(5):
            start_x = i * (width // 5)
            cv2.line(frame, (start_x, height), (vanish_x, vanish_y), (200, 200, 200), 1)
        
        # Add some noise to make it more realistic
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    print(f"Sample video created: {output_path}")

def create_fake_sample_video(output_path, duration_seconds=5, fps=30):
    """Create a sample video with geometric inconsistencies (simulating deepfake artifacts)"""
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    print(f"Creating fake sample video: {output_path}")
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background
        for y in range(height):
            intensity = int(50 + (y / height) * 100)
            frame[y, :] = [intensity, intensity + 20, intensity + 40]
        
        time_factor = frame_num / total_frames
        
        # Add object with INCONSISTENT shadow direction (fake artifact)
        rect_x = int(50 + time_factor * 400)
        rect_y = int(100 + 50 * np.sin(time_factor * 2 * np.pi))
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 100, rect_y + 80), 
                     (100, 150, 200), -1)
        
        # Inconsistent shadow - changes direction randomly
        shadow_dir = 1 if (frame_num // 10) % 2 == 0 else -1
        shadow_offset_x = shadow_dir * 20
        shadow_offset_y = 15
        cv2.rectangle(frame, 
                     (rect_x + shadow_offset_x, rect_y + shadow_offset_y), 
                     (rect_x + 100 + shadow_offset_x, rect_y + 80 + shadow_offset_y), 
                     (30, 30, 30), -1)
        
        # Add lines with INCONSISTENT vanishing points (fake artifact)
        # Use different vanishing points for different sets of lines
        vanish_x1 = int(width * 0.7 + 50 * np.sin(time_factor * 4 * np.pi))
        vanish_y1 = int(height * 0.3)
        vanish_x2 = int(width * 0.3)
        vanish_y2 = int(height * 0.7 + 30 * np.cos(time_factor * 3 * np.pi))
        
        # Lines converging to first vanishing point
        for i in range(3):
            start_x = i * (width // 3)
            cv2.line(frame, (start_x, height), (vanish_x1, vanish_y1), (200, 200, 200), 1)
        
        # Lines converging to second vanishing point (inconsistent)
        for i in range(2):
            start_x = (i + 3) * (width // 5)
            cv2.line(frame, (start_x, height), (vanish_x2, vanish_y2), (200, 200, 200), 1)
        
        # Add curved "lines" (should be straight)
        pts = np.array([
            [100, 300],
            [200, 295 + int(10 * np.sin(time_factor * 6 * np.pi))],
            [300, 290],
            [400, 285 + int(8 * np.cos(time_factor * 8 * np.pi))],
            [500, 280]
        ], np.int32)
        cv2.polylines(frame, [pts], False, (255, 255, 255), 2)
        
        # Add inconsistent lighting
        if frame_num % 15 < 7:  # Flickering light source
            overlay = frame.copy()
            cv2.circle(overlay, (width//4, height//4), 100, (255, 255, 200), -1)
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        out.write(frame)
    
    out.release()
    print(f"Fake sample video created: {output_path}")

def analyze_sample_videos():
    """Analyze the sample videos and compare results"""
    
    # Create sample videos if they don't exist
    real_video_path = "sample_real_video.mp4"
    fake_video_path = "sample_fake_video.mp4"
    
    if not os.path.exists(real_video_path):
        create_sample_video(real_video_path)
    
    if not os.path.exists(fake_video_path):
        create_fake_sample_video(fake_video_path)
    
    # Initialize detector
    detector = VideoProjectiveGeometryDetector(num_frames_to_analyze=8)
    
    print("\n" + "="*60)
    print("PROJECTIVE GEOMETRY DEEPFAKE DETECTION ANALYSIS")
    print("="*60)
    
    videos_to_analyze = [
        ("Real Video", real_video_path, False),
        ("Fake Video", fake_video_path, True)
    ]
    
    results = []
    
    for video_name, video_path, true_label in videos_to_analyze:
        print(f"\n--- Analyzing {video_name} ---")
        
        is_fake, confidence, detailed_results = detector.detect_deepfake_video(
            video_path, threshold=0.5
        )
        
        # Store results
        results.append({
            'name': video_name,
            'path': video_path,
            'predicted_fake': is_fake,
            'confidence': confidence,
            'true_label': true_label,
            'correct': (is_fake == true_label),
            'detailed': detailed_results
        })
        
        # Print results
        print(f"\nResults for {video_name}:")
        print(f"  Predicted: {'FAKE' if is_fake else 'REAL'}")
        print(f"  True Label: {'FAKE' if true_label else 'REAL'}")
        print(f"  Correct: {'✓' if (is_fake == true_label) else '✗'}")
        print(f"  Confidence Score: {confidence:.3f}")
        
        if 'avg_confidence' in detailed_results:
            print(f"  Average Score: {detailed_results['avg_confidence']:.3f}")
            print(f"  Max Score: {detailed_results['max_confidence']:.3f}")
            print(f"  Score Std: {detailed_results['std_confidence']:.3f}")
        
        # Show frame-by-frame analysis
        if 'frame_results' in detailed_results and detailed_results['frame_results']:
            print(f"\n  Frame-by-frame analysis:")
            for i, frame_result in enumerate(detailed_results['frame_results'][:3]):  # Show first 3
                scores = frame_result['scores']
                print(f"    Frame {frame_result['frame_index']}: "
                      f"P={scores['perspective_score']:.2f}, "
                      f"S={scores['shadow_score']:.2f}, "
                      f"L={scores['line_score']:.2f}, "
                      f"Combined={scores['combined_score']:.2f}")
    
    # Visualize results
    visualize_comparison_results(results, detector)
    
    return results

def visualize_comparison_results(results, detector):
    """Visualize comparison between real and fake video analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Overall confidence scores
    video_names = [r['name'] for r in results]
    confidences = [r['confidence'] for r in results]
    colors = ['green' if not r['predicted_fake'] else 'red' for r in results]
    
    axes[0, 0].bar(video_names, confidences, color=colors, alpha=0.7)
    axes[0, 0].set_title('Overall Confidence Scores')
    axes[0, 0].set_ylabel('Confidence (0=Real, 1=Fake)')
    axes[0, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Individual score components
    if len(results) >= 2 and 'frame_results' in results[0]['detailed']:
        real_scores = []
        fake_scores = []
        
        for result in results:
            if 'frame_results' in result['detailed']:
                frame_scores = [fr['scores'] for fr in result['detailed']['frame_results']]
                avg_scores = {
                    'perspective': np.mean([fs['perspective_score'] for fs in frame_scores]),
                    'shadow': np.mean([fs['shadow_score'] for fs in frame_scores]),
                    'line': np.mean([fs['line_score'] for fs in frame_scores])
                }
                
                if result['true_label']:  # Fake
                    fake_scores.append(avg_scores)
                else:  # Real
                    real_scores.append(avg_scores)
        
        if real_scores and fake_scores:
            categories = ['Perspective', 'Shadow', 'Line']
            real_vals = [real_scores[0]['perspective'], real_scores[0]['shadow'], real_scores[0]['line']]
            fake_vals = [fake_scores[0]['perspective'], fake_scores[0]['shadow'], fake_scores[0]['line']]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[0, 1].bar(x - width/2, real_vals, width, label='Real Video', color='green', alpha=0.7)
            axes[0, 1].bar(x + width/2, fake_vals, width, label='Fake Video', color='red', alpha=0.7)
            axes[0, 1].set_title('Score Components Comparison')
            axes[0, 1].set_ylabel('Inconsistency Score')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(categories)
            axes[0, 1].legend()
            axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Score distribution over frames
    for i, result in enumerate(results):
        if 'all_scores' in result['detailed']:
            all_scores = result['detailed']['all_scores']
            frame_indices = list(range(len(all_scores)))
            
            color = 'red' if result['true_label'] else 'green'
            label = f"{result['name']} ({'Fake' if result['true_label'] else 'Real'})"
            
            axes[1, 0].plot(frame_indices, all_scores, 
                           marker='o', color=color, alpha=0.7, label=label)
    
    axes[1, 0].set_title('Score Distribution Across Frames')
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('Combined Score')
    axes[1, 0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    axes[1, 1].text(0.1, 0.8, "Analysis Summary:", fontsize=14, fontweight='bold')
    
    for i, result in enumerate(results):
        y_pos = 0.6 - i * 0.3
        status = "✓ Correct" if result['correct'] else "✗ Incorrect"
        color = 'green' if result['correct'] else 'red'
        
        axes[1, 1].text(0.1, y_pos, f"{result['name']}:", fontsize=12, fontweight='bold')
        axes[1, 1].text(0.1, y_pos - 0.05, f"  Prediction: {'FAKE' if result['predicted_fake'] else 'REAL'}", fontsize=10)
        axes[1, 1].text(0.1, y_pos - 0.1, f"  {status}", fontsize=10, color=color)
        axes[1, 1].text(0.1, y_pos - 0.15, f"  Confidence: {result['confidence']:.3f}", fontsize=10)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Projective Geometry Deepfake Detection Results", fontsize=16, y=1.02)
    plt.show()

def analyze_single_frame_demo():
    """Demonstrate single frame analysis with visualization"""
    
    print("\n" + "="*50)
    print("SINGLE FRAME ANALYSIS DEMO")
    print("="*50)
    
    # Create or use existing sample video
    fake_video_path = "sample_fake_video.mp4"
    if not os.path.exists(fake_video_path):
        create_fake_sample_video(fake_video_path)
    
    # Initialize detector
    detector = VideoProjectiveGeometryDetector()
    
    # Extract one frame for detailed analysis
    frames, frame_indices = detector.extract_random_frames(fake_video_path, num_frames=1)
    
    if frames:
        frame = frames[0]
        print(f"Analyzing frame {frame_indices[0]} from {fake_video_path}")
        
        # Analyze the frame
        scores = detector.analyze_frame(frame)
        
        print(f"\nAnalysis Results:")
        print(f"  Perspective Score: {scores['perspective_score']:.3f}")
        print(f"  Shadow Score: {scores['shadow_score']:.3f}")
        print(f"  Line Score: {scores['line_score']:.3f}")
        print(f"  Combined Score: {scores['combined_score']:.3f}")
        print(f"  Prediction: {'FAKE' if scores['combined_score'] > 0.5 else 'REAL'}")
        
        # Visualize the analysis
        detector.visualize_frame_analysis(frame, scores, save_path="frame_analysis.png")
        print(f"\nVisualization saved as 'frame_analysis.png'")

def main():
    """Main function to run all demos"""
    
    print("🎬 Video Projective Geometry Deepfake Detector Demo")
    print("Based on 'Shadows Don't Lie and Lines Can't Bend!' (CVPR 2024)")
    print("Adapted for video analysis by extracting and analyzing random frames")
    
    try:
        # Run complete analysis on sample videos
        print("\n🔍 Running complete video analysis...")
        results = analyze_sample_videos()
        
        # Run single frame demo
        print("\n🖼️  Running single frame analysis demo...")
        analyze_single_frame_demo()
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        total_correct = sum(1 for r in results if r['correct'])
        accuracy = total_correct / len(results) if results else 0
        
        print(f"Total videos analyzed: {len(results)}")
        print(f"Correct predictions: {total_correct}")
        print(f"Accuracy: {accuracy:.1%}")
        
        print(f"\nMethod Details:")
        print(f"  • Perspective Field Analysis: Detects vanishing point inconsistencies")
        print(f"  • Shadow Consistency Analysis: Checks object-shadow relationships") 
        print(f"  • Line Analysis: Ensures lines remain straight (don't bend)")
        print(f"  • Combined Score: Average of all three geometric analyses")
        
        print(f"\nFiles created:")
        print(f"  • sample_real_video.mp4 - Synthetic video with consistent geometry")
        print(f"  • sample_fake_video.mp4 - Synthetic video with geometric inconsistencies")
        print(f"  • frame_analysis.png - Detailed frame analysis visualization")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Make sure OpenCV is installed: pip install opencv-python")
        print("Make sure other dependencies are installed: pip install numpy matplotlib tqdm")

if __name__ == "__main__":
    main() 