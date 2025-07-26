#!/usr/bin/env python3
"""
Create test videos for the Video Projective Geometry Deepfake Detector
This script generates various types of test videos with different characteristics
"""

import cv2
import numpy as np
import os
import random
import math
from tqdm import tqdm

def create_realistic_real_video(output_path, duration=10, fps=30):
    """Create a realistic video with consistent geometry (simulating real content)"""
    
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    print(f"Creating realistic 'REAL' video: {output_path}")
    
    for frame_num in tqdm(range(total_frames), desc="Generating frames"):
        # Create base scene
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky gradient
        for y in range(height//3):
            intensity = int(135 + (y / (height//3)) * 60)
            frame[y, :] = [intensity + 20, intensity + 10, intensity - 10]
        
        # Ground
        for y in range(height//3, height):
            intensity = int(80 + ((y - height//3) / (2*height//3)) * 40)
            frame[y, :] = [intensity - 20, intensity + 10, intensity - 30]
        
        time_factor = frame_num / total_frames
        
        # Add buildings with consistent perspective
        vanishing_x, vanishing_y = width * 0.6, height * 0.4
        
        # Building 1
        building_bottom = height - 100
        building_top = int(height * 0.3 + 20 * math.sin(time_factor * 2 * math.pi))
        
        # Left side of building
        cv2.line(frame, (200, building_bottom), (int(vanishing_x - 200), building_top), (100, 100, 100), 3)
        # Right side of building  
        cv2.line(frame, (400, building_bottom), (int(vanishing_x - 100), building_top), (100, 100, 100), 3)
        # Top line
        cv2.line(frame, (int(vanishing_x - 200), building_top), (int(vanishing_x - 100), building_top), (100, 100, 100), 3)
        
        # Fill building
        pts = np.array([
            [200, building_bottom],
            [int(vanishing_x - 200), building_top],
            [int(vanishing_x - 100), building_top],
            [400, building_bottom]
        ], np.int32)
        cv2.fillPoly(frame, [pts], (120, 120, 140))
        
        # Add consistent shadows
        shadow_offset_x = 30
        shadow_offset_y = 20
        shadow_pts = pts.copy()
        shadow_pts[:, 0] += shadow_offset_x
        shadow_pts[:, 1] += shadow_offset_y
        cv2.fillPoly(frame, [shadow_pts], (60, 60, 70))
        
        # Add moving car with consistent shadow
        car_x = int(100 + time_factor * 800)
        car_y = height - 200
        car_w, car_h = 80, 40
        
        # Car body
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (180, 60, 60), -1)
        
        # Car shadow (consistent direction)
        shadow_car_x = car_x + 25
        shadow_car_y = car_y + 35
        cv2.rectangle(frame, (shadow_car_x, shadow_car_y), 
                     (shadow_car_x + car_w, shadow_car_y + 15), (40, 40, 40), -1)
        
        # Add road lines (perspective correct)
        for i in range(5):
            line_y = height - 50 - i * 30
            line_start_x = int(50 + i * 20)
            line_end_x = int(width - 50 - i * 20)
            cv2.line(frame, (line_start_x, line_y), (line_end_x, line_y), (255, 255, 255), 2)
        
        # Add some noise for realism
        noise = np.random.randint(0, 15, (height, width, 3), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.95, noise, 0.05, 0)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Real video created: {output_path}")

def create_realistic_fake_video(output_path, duration=10, fps=30):
    """Create a video with geometric inconsistencies (simulating deepfake artifacts)"""
    
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    print(f"Creating realistic 'FAKE' video: {output_path}")
    
    for frame_num in tqdm(range(total_frames), desc="Generating frames"):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Similar base scene
        for y in range(height//3):
            intensity = int(135 + (y / (height//3)) * 60)
            frame[y, :] = [intensity + 20, intensity + 10, intensity - 10]
        
        for y in range(height//3, height):
            intensity = int(80 + ((y - height//3) / (2*height//3)) * 40)
            frame[y, :] = [intensity - 20, intensity + 10, intensity - 30]
        
        time_factor = frame_num / total_frames
        
        # INCONSISTENT vanishing points (fake artifact)
        vanishing_x1 = width * 0.6 + 100 * math.sin(time_factor * 4 * math.pi)
        vanishing_y1 = height * 0.4
        vanishing_x2 = width * 0.4
        vanishing_y2 = height * 0.6 + 50 * math.cos(time_factor * 3 * math.pi)
        
        # Building with inconsistent perspective
        building_bottom = height - 100
        building_top = int(height * 0.3)
        
        # Lines converging to different vanishing points (inconsistent)
        cv2.line(frame, (200, building_bottom), (int(vanishing_x1 - 200), building_top), (100, 100, 100), 3)
        cv2.line(frame, (400, building_bottom), (int(vanishing_x2 - 100), building_top), (100, 100, 100), 3)
        
        # Add curved "straight" lines (impossible in reality)
        pts_curved = []
        for i in range(20):
            t = i / 19.0
            x = int(100 + t * 600)
            y = int(height - 150 + 30 * math.sin(t * math.pi + time_factor * 2 * math.pi))
            pts_curved.append([x, y])
        
        pts_curved = np.array(pts_curved, np.int32)
        cv2.polylines(frame, [pts_curved], False, (200, 200, 200), 3)
        
        # Moving object with INCONSISTENT shadow direction
        car_x = int(100 + time_factor * 800)
        car_y = height - 200
        car_w, car_h = 80, 40
        
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (180, 60, 60), -1)
        
        # Shadow direction changes randomly (inconsistent lighting)
        shadow_direction = 1 if (frame_num // 20) % 2 == 0 else -1
        shadow_car_x = car_x + shadow_direction * 25
        shadow_car_y = car_y + 35
        cv2.rectangle(frame, (shadow_car_x, shadow_car_y), 
                     (shadow_car_x + car_w, shadow_car_y + 15), (40, 40, 40), -1)
        
        # Add flickering lighting (inconsistent illumination)
        if frame_num % 25 < 10:
            overlay = frame.copy()
            cv2.circle(overlay, (width//4, height//4), 150, (255, 255, 200), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add impossible reflections
        if frame_num % 40 < 20:
            reflection_y = height - 100
            cv2.rectangle(frame, (car_x, reflection_y), 
                         (car_x + car_w, reflection_y - car_h), (90, 30, 30), -1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Fake video created: {output_path}")

def create_face_like_video(output_path, duration=8, fps=30):
    """Create a video with face-like content for more realistic testing"""
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    print(f"Creating face-like video: {output_path}")
    
    for frame_num in tqdm(range(total_frames), desc="Generating frames"):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create skin-tone background
        frame[:, :] = [180, 160, 140]
        
        time_factor = frame_num / total_frames
        
        # Face oval
        center_x, center_y = width//2, height//2
        face_w, face_h = 200, 250
        cv2.ellipse(frame, (center_x, center_y), (face_w//2, face_h//2), 0, 0, 360, (200, 180, 160), -1)
        
        # Eyes
        eye_y = center_y - 40
        left_eye_x = center_x - 40
        right_eye_x = center_x + 40
        
        cv2.circle(frame, (left_eye_x, eye_y), 15, (50, 50, 50), -1)
        cv2.circle(frame, (right_eye_x, eye_y), 15, (50, 50, 50), -1)
        
        # Add subtle animation
        blink_offset = int(5 * math.sin(time_factor * 8 * math.pi))
        cv2.circle(frame, (left_eye_x, eye_y + blink_offset), 3, (255, 255, 255), -1)
        cv2.circle(frame, (right_eye_x, eye_y + blink_offset), 3, (255, 255, 255), -1)
        
        # Nose
        nose_pts = np.array([
            [center_x - 5, center_y - 10],
            [center_x + 5, center_y - 10],
            [center_x, center_y + 10]
        ], np.int32)
        cv2.fillPoly(frame, [nose_pts], (180, 160, 140))
        
        # Mouth
        mouth_y = center_y + 40
        cv2.ellipse(frame, (center_x, mouth_y), (25, 8), 0, 0, 180, (100, 80, 80), -1)
        
        # Add lighting inconsistency for "fake" characteristics
        if "fake" in output_path.lower():
            # Inconsistent lighting direction
            light_x = int(width * 0.3 + width * 0.4 * math.sin(time_factor * 3 * math.pi))
            light_y = int(height * 0.2)
            
            # Create inconsistent shadow
            shadow_offset_x = int(20 * math.cos(time_factor * 4 * math.pi))
            shadow_offset_y = 15
            
            # Face shadow (inconsistent direction)
            cv2.ellipse(frame, (center_x + shadow_offset_x, center_y + shadow_offset_y), 
                       (face_w//2, face_h//2), 0, 0, 360, (160, 140, 120), -1)
        else:
            # Consistent lighting
            shadow_offset_x = 15
            shadow_offset_y = 15
            cv2.ellipse(frame, (center_x + shadow_offset_x, center_y + shadow_offset_y), 
                       (face_w//2, face_h//2), 0, 0, 360, (160, 140, 120), -1)
        
        # Add some texture
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.9, noise, 0.1, 0)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Face-like video created: {output_path}")

def create_all_test_videos():
    """Create a comprehensive set of test videos"""
    
    print("🎬 Creating comprehensive test video suite...")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("test_videos", exist_ok=True)
    
    # Create different types of test videos
    test_videos = [
        ("test_videos/realistic_real_scene.mp4", create_realistic_real_video, "Real scene with consistent geometry"),
        ("test_videos/realistic_fake_scene.mp4", create_realistic_fake_video, "Fake scene with geometric inconsistencies"),
        ("test_videos/face_like_real.mp4", lambda path, **kwargs: create_face_like_video(path, **kwargs), "Face-like content (real)"),
        ("test_videos/face_like_fake.mp4", lambda path, **kwargs: create_face_like_video(path.replace("real", "fake"), **kwargs), "Face-like content (fake)"),
    ]
    
    created_videos = []
    
    for video_path, create_func, description in test_videos:
        print(f"\n📹 {description}")
        try:
            create_func(video_path, duration=8, fps=24)  # Shorter for faster testing
            created_videos.append((video_path, description))
        except Exception as e:
            print(f"❌ Error creating {video_path}: {str(e)}")
    
    return created_videos

def test_created_videos():
    """Test the created videos with our detector"""
    
    print(f"\n🔍 Testing created videos with detector...")
    print("=" * 60)
    
    try:
        from video_deepfake_detector import VideoProjectiveGeometryDetector
        
        detector = VideoProjectiveGeometryDetector(num_frames_to_analyze=5)
        
        # Test all videos in test_videos directory
        video_files = []
        if os.path.exists("test_videos"):
            for filename in os.listdir("test_videos"):
                if filename.endswith(".mp4"):
                    video_files.append(os.path.join("test_videos", filename))
        
        if not video_files:
            print("❌ No test videos found. Run create_all_test_videos() first.")
            return
        
        results = []
        
        for video_path in video_files:
            print(f"\n🎬 Testing: {os.path.basename(video_path)}")
            
            try:
                is_fake, confidence, detailed_results = detector.detect_deepfake_video(video_path, threshold=0.5)
                
                results.append({
                    'video': os.path.basename(video_path),
                    'predicted_fake': is_fake,
                    'confidence': confidence,
                    'expected_fake': 'fake' in video_path.lower()
                })
                
                expected = 'fake' in video_path.lower()
                correct = (is_fake == expected)
                
                print(f"  Result: {'FAKE' if is_fake else 'REAL'} (confidence: {confidence:.3f})")
                print(f"  Expected: {'FAKE' if expected else 'REAL'}")
                print(f"  Status: {'✅ Correct' if correct else '❌ Incorrect'}")
                
            except Exception as e:
                print(f"  ❌ Error testing {video_path}: {str(e)}")
        
        # Summary
        print(f"\n📊 SUMMARY")
        print("=" * 40)
        
        if results:
            correct_predictions = sum(1 for r in results if r['predicted_fake'] == r['expected_fake'])
            accuracy = correct_predictions / len(results)
            
            print(f"Total videos tested: {len(results)}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Accuracy: {accuracy:.1%}")
            
            print(f"\nDetailed Results:")
            for r in results:
                status = "✅" if r['predicted_fake'] == r['expected_fake'] else "❌"
                print(f"  {status} {r['video']}: {r['confidence']:.3f} ({'FAKE' if r['predicted_fake'] else 'REAL'})")
        
        return results
        
    except ImportError:
        print("❌ Could not import video_deepfake_detector. Make sure it's in the same directory.")
        return []
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return []

def main():
    """Main function"""
    
    print("🎥 Test Video Generator for Deepfake Detector")
    print("Based on 'Shadows Don't Lie and Lines Can't Bend!' adaptation")
    print("=" * 60)
    
    # Create test videos
    created_videos = create_all_test_videos()
    
    if created_videos:
        print(f"\n✅ Successfully created {len(created_videos)} test videos:")
        for video_path, description in created_videos:
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            print(f"  📹 {os.path.basename(video_path)} ({file_size:.1f} MB) - {description}")
        
        # Test the videos
        print(f"\n🧪 Now testing the videos with our detector...")
        test_results = test_created_videos()
        
        print(f"\n🎯 USAGE INSTRUCTIONS:")
        print(f"  1. Use any of the created videos with: python test_your_video.py test_videos/video_name.mp4")
        print(f"  2. Or run complete analysis with: python sample_usage.py")
        print(f"  3. Videos are saved in 'test_videos/' directory")
        
        print(f"\n💡 TIP: These synthetic videos demonstrate the detector's ability to")
        print(f"  identify geometric inconsistencies. Real deepfake videos would have")
        print(f"  similar but more subtle geometric artifacts.")
        
    else:
        print("❌ No videos were created successfully.")

if __name__ == "__main__":
    main() 