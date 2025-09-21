#!/usr/bin/env python3
"""
Performance testing script for the Advanced Vehicle Detection System
"""

import time
import cv2
import numpy as np
import argparse
from advanced_vehicle_detector import AdvancedVehicleDetector
import json

def benchmark_detection_system(config_file=None, duration=60):
    """Benchmark the detection system performance."""
    print("🔬 Performance Benchmark Starting...")
    print(f"⏱️ Test Duration: {duration} seconds")
    
    # Initialize detector
    detector = AdvancedVehicleDetector(config_file)
    
    # Initialize camera
    cap = cv2.VideoCapture(detector.config['camera_index'])
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return None
    
    # Performance metrics
    frame_times = []
    detection_counts = []
    fps_history = []
    
    start_time = time.time()
    frame_count = 0
    
    print("🎬 Starting benchmark...")
    
    try:
        while time.time() - start_time < duration:
            frame_start = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            frame = cv2.resize(frame, (detector.config['resize_width'], 
                                     int(frame.shape[0] * detector.config['resize_width'] / frame.shape[1])))
            
            # Detect vehicles
            detections = detector.detect_vehicles_gpu(frame) if detector.gpu_enabled else detector.detect_vehicles_cpu(frame)
            
            # Update tracking
            tracked_objects = detector.update_tracking(detections)
            
            # Record metrics
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            detection_counts.append(len(tracked_objects))
            fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
            
            frame_count += 1
            
            # Display progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = np.mean(fps_history[-30:]) if fps_history else 0
                print(f"📊 Frame {frame_count}, Elapsed: {elapsed:.1f}s, FPS: {current_fps:.1f}")
                
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted")
    
    finally:
        cap.release()
    
    # Calculate final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_processing_time = np.mean(frame_times) * 1000 if frame_times else 0
    avg_detections = np.mean(detection_counts) if detection_counts else 0
    max_fps = max(fps_history) if fps_history else 0
    min_fps = min(fps_history) if fps_history else 0
    
    results = {
        'total_frames': frame_count,
        'total_time': total_time,
        'average_fps': avg_fps,
        'max_fps': max_fps,
        'min_fps': min_fps,
        'average_processing_time_ms': avg_processing_time,
        'average_detections_per_frame': avg_detections,
        'total_detections': sum(detection_counts),
        'gpu_enabled': detector.gpu_enabled,
        'config': detector.config
    }
    
    return results

def compare_configurations():
    """Compare different configuration settings."""
    print("🔍 Configuration Comparison Test")
    print("=" * 50)
    
    configs = {
        'High Accuracy': {
            'scale_factor': 1.05,
            'min_neighbors': 5,
            'confidence_threshold': 0.5,
            'resize_width': 800,
            'enable_gpu': True,
            'enable_tracking': True
        },
        'Balanced': {
            'scale_factor': 1.1,
            'min_neighbors': 3,
            'confidence_threshold': 0.3,
            'resize_width': 640,
            'enable_gpu': True,
            'enable_tracking': True
        },
        'High Speed': {
            'scale_factor': 1.2,
            'min_neighbors': 2,
            'confidence_threshold': 0.2,
            'resize_width': 320,
            'enable_gpu': True,
            'enable_tracking': False
        },
        'CPU Only': {
            'scale_factor': 1.1,
            'min_neighbors': 3,
            'confidence_threshold': 0.3,
            'resize_width': 640,
            'enable_gpu': False,
            'enable_tracking': True
        }
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n🧪 Testing {config_name} configuration...")
        
        # Create temporary config file
        temp_config = 'temp_config.json'
        with open(temp_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run benchmark
        result = benchmark_detection_system(temp_config, duration=30)
        if result:
            results[config_name] = result
            
        # Clean up
        import os
        if os.path.exists(temp_config):
            os.remove(temp_config)
    
    # Display comparison results
    print("\n📊 CONFIGURATION COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Configuration':<15} {'Avg FPS':<8} {'Max FPS':<8} {'Avg Time (ms)':<12} {'Detections':<10}")
    print("-" * 60)
    
    for config_name, result in results.items():
        print(f"{config_name:<15} {result['average_fps']:<8.1f} {result['max_fps']:<8.1f} "
              f"{result['average_processing_time_ms']:<12.1f} {result['average_detections_per_frame']:<10.1f}")
    
    return results

def print_detailed_results(results):
    """Print detailed benchmark results."""
    if not results:
        print("❌ No results to display")
        return
        
    print("\n📈 DETAILED PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"🎬 Total Frames Processed: {results['total_frames']}")
    print(f"⏱️ Total Test Duration: {results['total_time']:.2f} seconds")
    print(f"📊 Average FPS: {results['average_fps']:.2f}")
    print(f"🚀 Maximum FPS: {results['max_fps']:.2f}")
    print(f"🐌 Minimum FPS: {results['min_fps']:.2f}")
    print(f"⚡ Average Processing Time: {results['average_processing_time_ms']:.2f} ms")
    print(f"🚗 Average Detections per Frame: {results['average_detections_per_frame']:.2f}")
    print(f"📈 Total Detections: {results['total_detections']}")
    print(f"💻 GPU Acceleration: {'Enabled' if results['gpu_enabled'] else 'Disabled'}")
    
    # Performance assessment
    print("\n🎯 PERFORMANCE ASSESSMENT")
    print("-" * 30)
    
    if results['average_fps'] >= 30:
        print("✅ Excellent performance - Suitable for real-time applications")
    elif results['average_fps'] >= 20:
        print("✅ Good performance - Suitable for most applications")
    elif results['average_fps'] >= 15:
        print("⚠️ Acceptable performance - May need optimization")
    else:
        print("❌ Poor performance - Requires optimization")
        
    if results['average_processing_time_ms'] < 50:
        print("✅ Low latency - Excellent for real-time processing")
    elif results['average_processing_time_ms'] < 100:
        print("✅ Good latency - Suitable for real-time applications")
    else:
        print("⚠️ High latency - Consider optimization")

def main():
    """Main function for performance testing."""
    parser = argparse.ArgumentParser(description='Performance Testing for Vehicle Detection System')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds (default: 60)')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--compare', action='store_true', help='Compare different configurations')
    parser.add_argument('--save-results', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    if args.compare:
        results = compare_configurations()
    else:
        results = benchmark_detection_system(args.config, args.duration)
        if results:
            print_detailed_results(results)
    
    # Save results if requested
    if args.save_results and results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {args.save_results}")

if __name__ == "__main__":
    main()
