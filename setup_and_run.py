#!/usr/bin/env python3
"""
Setup and quick usage guide for Video Projective Geometry Deepfake Detector
"""

import sys
import subprocess
import importlib
import os

def check_dependencies():
    """Check if all required packages are installed"""
    
    required_packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
        ('sklearn', 'scikit-learn')
    ]
    
    missing_packages = []
    
    print("🔍 Checking dependencies...")
    
    for package_name, pip_name in required_packages:
        try:
            importlib.import_module(package_name)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} - MISSING")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    else:
        print(f"\n✅ All dependencies are installed!")
        return True

def run_demo():
    """Run the complete demo"""
    print("\n🎬 Running complete demo...")
    try:
        import sample_usage
        sample_usage.main()
    except ImportError:
        print("❌ Could not import sample_usage.py")
        print("Make sure sample_usage.py is in the same directory")
    except Exception as e:
        print(f"❌ Error running demo: {str(e)}")

def show_usage_examples():
    """Show usage examples"""
    
    print("\n" + "="*60)
    print("📋 USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. 🎬 Run Complete Demo (creates sample videos and analyzes them):")
    print("   python sample_usage.py")
    
    print("\n2. 🔍 Test Your Own Video:")
    print("   python test_your_video.py your_video.mp4")
    print("   python test_your_video.py your_video.mp4 10 0.6")
    
    print("\n3. 💻 Use in Your Code:")
    print("""
   from video_deepfake_detector import VideoProjectiveGeometryDetector
   
   # Initialize detector
   detector = VideoProjectiveGeometryDetector(num_frames_to_analyze=8)
   
   # Analyze video
   is_fake, confidence, results = detector.detect_deepfake_video("video.mp4")
   
   print(f"Prediction: {'FAKE' if is_fake else 'REAL'}")
   print(f"Confidence: {confidence:.3f}")
   """)
    
    print("\n4. 🖼️  Analyze Single Frame with Visualization:")
    print("""
   # Extract frame and analyze
   frames, _ = detector.extract_random_frames("video.mp4", num_frames=1)
   scores = detector.analyze_frame(frames[0])
   
   # Visualize results
   detector.visualize_frame_analysis(frames[0], scores)
   """)

def main():
    """Main setup and usage guide"""
    
    print("🎥 Video Projective Geometry Deepfake Detector")
    print("Based on 'Shadows Don't Lie and Lines Can't Bend!' (CVPR 2024)")
    print("Adapted for video analysis")
    print("="*60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies first:")
        print("   pip install -r requirements_video_detector.txt")
        print("   # OR install individually:")
        print("   pip install opencv-python numpy matplotlib tqdm scikit-learn")
        return
    
    # Check if main files exist
    required_files = [
        'video_deepfake_detector.py',
        'sample_usage.py', 
        'test_your_video.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n❌ Missing required files: {', '.join(missing_files)}")
        print("Make sure all project files are in the same directory")
        return
    
    print(f"\n✅ All files present!")
    
    # Show usage examples
    show_usage_examples()
    
    # Ask user what they want to do
    print("\n" + "="*60)
    print("🚀 QUICK START OPTIONS")
    print("="*60)
    
    while True:
        print(f"\nWhat would you like to do?")
        print(f"  1. Run complete demo (creates and analyzes sample videos)")
        print(f"  2. Test your own video file")
        print(f"  3. Just show usage examples again")
        print(f"  4. Exit")
        
        try:
            choice = input(f"\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                run_demo()
                break
            elif choice == '2':
                video_path = input("Enter path to your video file: ").strip()
                if os.path.exists(video_path):
                    print(f"\nRunning: python test_your_video.py {video_path}")
                    os.system(f'python test_your_video.py "{video_path}"')
                else:
                    print(f"❌ File not found: {video_path}")
                break
            elif choice == '3':
                show_usage_examples()
                continue
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print(f"\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            break

if __name__ == "__main__":
    main() 