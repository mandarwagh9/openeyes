#!/usr/bin/env python3
"""Demo script showcasing all OpenEyes DeepStream features.

Usage:
    python demo_all_features.py              # Interactive menu
    python demo_all_features.py 1         # Run demo 1
    python demo_all_features.py all      # Run all demos
"""

import subprocess
import sys
import os
import time

DEMOS = {
    "1": {
        "name": "Basic DeepStream Pipeline",
        "desc": "CSI camera with YOLO detection + display + FPS",
        "cmd": ["python", "-m", "src.main", "--deepstream", "--camera", "0"],
    },
    "2": {
        "name": "Headless (No Display)",
        "desc": "Run without display - useful for headless servers",
        "cmd": ["python", "-m", "src.main", "--deepstream", "--camera", "0", "--no-display"],
    },
    "3": {
        "name": "720p Resolution",
        "desc": "1280x720 for better detection quality",
        "cmd": ["python", "-m", "src.main", "--deepstream", "--camera", "0", "--width", "1280", "--height", "720"],
    },
    "4": {
        "name": "480p @ 60fps",
        "desc": "640x480 at 60 FPS for smooth tracking",
        "cmd": ["python", "-m", "src.main", "--deepstream", "--camera", "0", "--fps", "60"],
    },
    "5": {
        "name": "360p Low Res",
        "desc": "640x360 for maximum performance",
        "cmd": ["python", "-m", "src.main", "--deepstream", "--camera", "0", "--width", "640", "--height", "360"],
    },
    "6": {
        "name": "UDP Output Enabled",
        "desc": "Send JSON to 127.0.0.1:5000",
        "cmd": ["python", "-m", "src.main", "--deepstream", "--camera", "0"],
    },
    "7": {
        "name": "ROS2 Output",
        "desc": "Publish detections to ROS2 topics",
        "cmd": ["python", "-m", "src.main", "--deepstream", "--camera", "0", "--ros2"],
    },
    "8": {
        "name": "UDP + ROS2",
        "desc": "Both UDP and ROS2 enabled",
        "cmd": ["python", "-m", "src.main", "--deepstream", "--camera", "0", "--ros2"],
    },
    "9": {
        "name": "Multi-Camera",
        "desc": "Use multi-camera config",
        "cmd": ["python", "-m", "src.main", "--deepstream", "--multi-camera", "0", "1"],
    },
    "10": {
        "name": "DLA Accelerator",
        "desc": "Use Deep Learning Accelerator",
        "cmd": ["python", "-m", "src.main", "--deepstream", "--camera", "0", "--dla"],
    },
    "11": {
        "name": "Debug Mode",
        "desc": "Verbose debug logging",
        "cmd": ["python", "-m", "src.main", "--deepstream", "--camera", "0", "--debug"],
    },
    "12": {
        "name": "Record Video",
        "desc": "Save to output.mp4",
        "cmd": ["python", "-m", "src.main", "--output", "output.mp4", "--camera", "0"],
    },
}


def print_menu():
    print("\n" + "=" * 60)
    print("🖥️  OpenEyes DeepStream Demo - All Features")
    print("=" * 60)
    print("\nSelect a demo to run:\n")
    
    for num, demo in DEMOS.items():
        print(f"  [{num}] {demo['name']}")
        print(f"      → {demo['desc']}")
        print()
    
    print("  [all] Run all demos sequentially (5 sec each)")
    print("  [q]  Quit")
    print("\n" + "-" * 60)


def run_demo(demo_id):
    demo = DEMOS.get(demo_id)
    if not demo:
        print(f"❌ Demo {demo_id} not found")
        return
    
    print(f"\n🚀 Starting: {demo['name']}")
    print(f"   {demo['desc']}")
    print(f"   Command: {' '.join(demo['cmd'])}")
    print("-" * 60)
    
    try:
        subprocess.run(demo['cmd'], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")


def run_all_demos():
    print("\n🔄 Running all demos sequentially...")
    print("⏱️  Each demo runs for 5 seconds")
    print("💡 Press Ctrl+C to stop at any time\n")
    
    time.sleep(2)
    
    for num, demo in DEMOS.items():
        print(f"\n{'='*60}")
        print(f"Demo {num}: {demo['name']}")
        print(f"{'='*60}")
        
        try:
            proc = subprocess.Popen(demo['cmd'])
            time.sleep(5)
            proc.terminate()
            proc.wait(timeout=2)
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
            break
        except subprocess.TimeoutExpired:
            proc.kill()
    
    print("\n✅ All demos completed!")


def main():
    os.chdir("/home/mandar/openeyes")
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "all":
            run_all_demos()
        elif arg in DEMOS:
            run_demo(arg)
        elif arg == "q":
            sys.exit(0)
        else:
            print(f"Unknown demo: {arg}")
            print_menu()
    else:
        print_menu()
        
        while True:
            try:
                choice = input("\n> ").strip().lower()
                
                if choice == "q":
                    print("👋 Goodbye!")
                    break
                elif choice == "all":
                    run_all_demos()
                    print_menu()
                elif choice in DEMOS:
                    run_demo(choice)
                else:
                    print("Invalid choice. Try 1-12 or 'all'")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                break


if __name__ == "__main__":
    main()