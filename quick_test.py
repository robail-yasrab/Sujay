#!/usr/bin/env python3
"""
Quick test script - creates a sample video and tests it immediately
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

def create_quick_test_video(output_path="quick_test_video.mp4"):
    """Create a quick test video with geometric inconsistencies"""
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    print(f"🎬 Creating quick test video: {output_path}")
    
    # Create 120 frames (4 seconds at 30fps)
    for frame_num in tqdm(range(120), desc="Creating frames"):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            intensity = int(50 + (y / height) * 100)
            frame[y, :] = [intensity, intensity + 20, intensity + 40]
        
        time_factor = frame_num / 120
        
        # Add a moving object
        obj_x = int(50 + time_factor * 400)
        obj_y = 200
        cv2.rectangle(frame, (obj_x, obj_y), (obj_x + 80, obj_y + 60), (150, 100, 200), -1)
        
        # Add INCONSISTENT shadow (changes direction - fake characteristic)
        shadow_direction = 1 if frame_num % 30 < 15 else -1
        shadow_x = obj_x + shadow_direction * 30
        shadow_y = obj_y + 50
        cv2.rectangle(frame, (shadow_x, shadow_y), (shadow_x + 80, shadow_y + 20), (30, 30, 30), -1)
        
        # Add some lines with INCONSISTENT vanishing points
        # First set of lines
        vp1_x = width * 0.7
        vp1_y = height * 0.3
        cv2.line(frame, (0, height), (int(vp1_x), int(vp1_y)), (255, 255, 255), 2)
        cv2.line(frame, (width//3, height), (int(vp1_x), int(vp1_y)), (255, 255, 255), 2)
        
        # Second set with DIFFERENT vanishing point (inconsistent!)
        vp2_x = width * 0.3 + 50 * np.sin(time_factor * 4 * np.pi)
        vp2_y = height * 0.7
        cv2.line(frame, (2*width//3, height), (int(vp2_x), int(vp2_y)), (255, 255, 255), 2)
        cv2.line(frame, (width, height), (int(vp2_x), int(vp2_y)), (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Video created: {output_path}")
    return output_path

def test_video_immediately(video_path):
    """Test the video immediately after creation"""
    
    print(f"\n🔍 Testing video with deepfake detector...")
    
    try:
        from video_deepfake_detector import VideoProjectiveGeometryDetector
        
        # Initialize detector
        detector = VideoProjectiveGeometryDetector(num_frames_to_analyze=6)
        
        # Analyze the video
        is_fake, confidence, results = detector.detect_deepfake_video(video_path, threshold=0.5)
        
        print(f"\n📊 RESULTS:")
        print(f"   🎯 Prediction: {'🚨 FAKE' if is_fake else '✅ REAL'}")
        print(f"   📈 Confidence: {confidence:.3f}")
        print(f"   🎲 Expected: FAKE (due to inconsistent shadows & vanishing points)")
        
        if 'frame_results' in results:
            print(f"\n📋 Frame Analysis:")
            for i, frame_result in enumerate(results['frame_results'][:3]):
                scores = frame_result['scores']
                print(f"   Frame {i+1}: P={scores['perspective_score']:.2f} "
                      f"S={scores['shadow_score']:.2f} L={scores['line_score']:.2f} "
                      f"→ {scores['combined_score']:.2f}")
        
        # Determine if the test was successful
        success = is_fake  # We expect it to be detected as fake
        
        print(f"\n🎉 Test Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if success:
            print("   The detector correctly identified geometric inconsistencies!")
        else:
            print("   The detector didn't detect the inconsistencies. Try adjusting the threshold.")
        
        return success
        
    except ImportError:
        print("❌ Error: Could not import video_deepfake_detector.")
        print("   Make sure video_deepfake_detector.py is in the same directory.")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def main():
    """Main function"""
    
    print("⚡ Quick Video Test for Projective Geometry Deepfake Detector")
    print("=" * 60)
    
    # Create test video
    video_path = create_quick_test_video()
    
    # Get file size
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    print(f"📁 File size: {file_size:.1f} MB")
    
    # Test immediately
    success = test_video_immediately(video_path)
    
    print(f"\n🎯 NEXT STEPS:")
    
    if success:
        print("   1. ✅ The detector is working correctly!")
        print("   2. 📺 Test with your own videos:")
        print(f"      python test_your_video.py {video_path}")
        print("   3. 🎬 Create more test videos:")
        print("      python create_test_videos.py")
        print("   4. 🔬 Run full demo:")
        print("      python sample_usage.py")
    else:
        print("   1. 🔧 Check that all files are in the same directory")
        print("   2. 📦 Install dependencies:")
        print("      pip install opencv-python numpy matplotlib tqdm")
        print("   3. 🧪 Try running setup:")
        print("      python setup_and_run.py")
    
    print(f"\n📹 Your test video is saved as: {video_path}")
    print("   You can view it with any video player to see the geometric inconsistencies.")

if __name__ == "__main__":
    main() 