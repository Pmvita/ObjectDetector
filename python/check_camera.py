#!/usr/bin/env python3
"""
Camera permission checker for macOS.
Helps diagnose camera access issues.
"""

import cv2
import sys
import platform

def check_camera_access():
    """Check if camera is accessible."""
    print("=" * 60)
    print("Camera Access Checker")
    print("=" * 60)
    print()
    
    if platform.system() != 'Darwin':
        print("Note: This script is optimized for macOS.")
        print("On other platforms, camera access should work normally.")
        print()
    
    print("Checking camera access...")
    print()
    
    # Try to open camera
    for camera_idx in range(3):
        print(f"Trying camera {camera_idx}...")
        cap = cv2.VideoCapture(camera_idx)
        
        if not cap.isOpened():
            print(f"  ✗ Camera {camera_idx}: Could not open")
            continue
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret and frame is not None:
            height, width = frame.shape[:2]
            print(f"  ✓ Camera {camera_idx}: Working!")
            print(f"    Resolution: {width}x{height}")
            cap.release()
            return camera_idx
        else:
            print(f"  ✗ Camera {camera_idx}: Opened but cannot read frames")
            cap.release()
    
    print()
    print("=" * 60)
    print("Camera Access Issues Detected")
    print("=" * 60)
    print()
    
    if platform.system() == 'Darwin':
        print("macOS Camera Permission Fix:")
        print()
        print("1. Open System Preferences (or System Settings on macOS 13+)")
        print("2. Go to Security & Privacy > Privacy > Camera")
        print("3. Find 'Terminal' or 'Python' in the list")
        print("4. Check the box to allow camera access")
        print()
        print("If Terminal/Python is not in the list:")
        print("  - Try running the detector once, then check again")
        print("  - Or grant permissions via command line:")
        print("    tccutil reset Camera")
        print()
        print("Alternative: Use a different terminal app that has camera access")
        print("  - iTerm2 (may need permission)")
        print("  - Or run from an IDE like VS Code/PyCharm")
        print()
    
    print("Other things to check:")
    print("  - Close other apps using the camera (Zoom, FaceTime, Photo Booth, etc.)")
    print("  - Restart your terminal/IDE")
    print("  - Try restarting your computer")
    print()
    
    return None

if __name__ == "__main__":
    working_camera = check_camera_access()
    if working_camera is not None:
        print(f"\n✓ Camera {working_camera} is working!")
        print(f"Run: python3 detector.py --camera {working_camera}")
        sys.exit(0)
    else:
        print("\n✗ No working camera found")
        sys.exit(1)

