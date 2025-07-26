#!/usr/bin/env python3
"""
Simple script to test the Video Projective Geometry Deepfake Detector on your own videos
"""

import os
import sys
from video_deepfake_detector import VideoProjectiveGeometryDetector

def test_video(video_path, num_frames=8, threshold=0.6):
    """Test a single video file"""
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' not found!")
        return False
    
    print(f"🎬 Testing video: {os.path.basename(video_path)}")
    print(f"📊 Will analyze {num_frames} random frames")
    print(f"🎯 Detection threshold: {threshold}")
    print("-" * 50)
    
    # Initialize detector
    detector = VideoProjectiveGeometryDetector(num_frames_to_analyze=num_frames)
    
    try:
        # Analyze the video
        is_fake, confidence, results = detector.detect_deepfake_video(
            video_path, threshold=threshold
        )
        
        # Print results
        print(f"\n🔍 ANALYSIS RESULTS:")
        print(f"   Prediction: {'🚨 FAKE' if is_fake else '✅ REAL'}")
        print(f"   Confidence Score: {confidence:.3f}")
        
        if 'avg_confidence' in results:
            print(f"   Average Score: {results['avg_confidence']:.3f}")
            print(f"   Max Score: {results['max_confidence']:.3f}")
            print(f"   Score Variation: {results['std_confidence']:.3f}")
        
        # Show detailed frame analysis
        if 'frame_results' in results and results['frame_results']:
            print(f"\n📋 FRAME-BY-FRAME ANALYSIS:")
            print(f"   Frame | Perspective | Shadow | Line | Combined")
            print(f"   ------|-------------|--------|------|----------")
            
            for frame_result in results['frame_results'][:5]:  # Show first 5 frames
                scores = frame_result['scores']
                frame_idx = frame_result['frame_index']
                print(f"   {frame_idx:5d} | {scores['perspective_score']:11.3f} | "
                      f"{scores['shadow_score']:6.3f} | {scores['line_score']:4.3f} | "
                      f"{scores['combined_score']:8.3f}")
            
            if len(results['frame_results']) > 5:
                print(f"   ... and {len(results['frame_results']) - 5} more frames")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        if confidence < 0.3:
            print("   → Strong indication of REAL content")
        elif confidence < 0.5:
            print("   → Likely REAL content")
        elif confidence < 0.7:
            print("   → Possibly FAKE content")
        else:
            print("   → Strong indication of FAKE content")
        
        print(f"\n📝 ANALYSIS DETAILS:")
        print(f"   • Perspective Score: Measures vanishing point consistency")
        print(f"   • Shadow Score: Checks object-shadow relationships")
        print(f"   • Line Score: Ensures lines remain straight")
        print(f"   • Combined Score: Average of all three analyses")
        print(f"   • Higher scores (>0.6) suggest deepfake artifacts")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing video: {str(e)}")
        return False

def main():
    """Main function"""
    
    print("🎥 Video Projective Geometry Deepfake Detector")
    print("Based on 'Shadows Don't Lie and Lines Can't Bend!' (CVPR 2024)")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_your_video.py <video_path> [num_frames] [threshold]")
        print("\nExamples:")
        print("  python test_your_video.py my_video.mp4")
        print("  python test_your_video.py my_video.mp4 10")
        print("  python test_your_video.py my_video.mp4 10 0.5")
        print("\nParameters:")
        print("  video_path  : Path to your video file")
        print("  num_frames  : Number of random frames to analyze (default: 8)")
        print("  threshold   : Detection threshold (default: 0.6)")
        return
    
    # Parse arguments
    video_path = sys.argv[1]
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    
    # Validate parameters
    if num_frames < 1 or num_frames > 50:
        print(f"❌ Error: num_frames should be between 1 and 50, got {num_frames}")
        return
    
    if threshold < 0.0 or threshold > 1.0:
        print(f"❌ Error: threshold should be between 0.0 and 1.0, got {threshold}")
        return
    
    # Test the video
    success = test_video(video_path, num_frames, threshold)
    
    if success:
        print(f"\n✅ Analysis completed successfully!")
        print(f"\n💭 Note: This detector analyzes geometric consistency.")
        print(f"   Real videos should have consistent perspective, shadows, and straight lines.")
        print(f"   Deepfakes often have geometric inconsistencies due to generation artifacts.")
    else:
        print(f"\n❌ Analysis failed. Please check your video file and try again.")
        print(f"\n🔧 Troubleshooting:")
        print(f"   • Make sure the video file exists and is readable")
        print(f"   • Supported formats: mp4, avi, mov, mkv, etc.")
        print(f"   • Install required packages: pip install -r requirements_video_detector.txt")

if __name__ == "__main__":
    main() 