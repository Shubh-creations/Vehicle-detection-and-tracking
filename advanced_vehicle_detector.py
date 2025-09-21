#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Vehicle Detection and Tracking System
==============================================

A real-time vehicle detection and tracking system using OpenCV with GPU acceleration,
advanced tracking algorithms, and comprehensive traffic analytics.

Features:
- GPU-accelerated inference (CUDA)
- Advanced vehicle tracking to prevent double counting
- Real-time FPS monitoring and performance metrics
- Enhanced visualization with traffic analytics
- Configurable parameters for different scenarios
- Edge device optimization

Author: Advanced Vehicle Detection System
Version: 2.0
"""

import cv2
import numpy as np
import time
import json
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import imutils
import os
from datetime import datetime

@dataclass
class Vehicle:
    """Represents a tracked vehicle with its properties."""
    id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    last_seen: float
    track_history: List[Tuple[int, int]]
    confidence: float
    direction: str = "unknown"
    
@dataclass
class TrafficStats:
    """Traffic statistics container."""
    total_vehicles: int = 0
    vehicles_per_minute: float = 0.0
    avg_confidence: float = 0.0
    fps: float = 0.0
    processing_time: float = 0.0

class VehicleTracker:
    """Advanced vehicle tracking system using centroid tracking."""
    
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, bbox, confidence):
        """Register a new vehicle."""
        self.objects[self.next_object_id] = Vehicle(
            id=self.next_object_id,
            centroid=centroid,
            bbox=bbox,
            last_seen=time.time(),
            track_history=[centroid],
            confidence=confidence
        )
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1
        
    def deregister(self, object_id):
        """Remove a vehicle from tracking."""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
            
    def update(self, detections):
        """Update vehicle tracking with new detections."""
        if len(detections) == 0:
            # No detections, mark all as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
            
        # Compute centroids for new detections
        input_centroids = []
        input_bboxes = []
        input_confidences = []
        
        for (x, y, w, h, conf) in detections:
            centroid_x = int(x + w // 2)
            centroid_y = int(y + h // 2)
            input_centroids.append((centroid_x, centroid_y))
            input_bboxes.append((x, y, w, h))
            input_confidences.append(conf)
            
        if len(self.objects) == 0:
            # No existing objects, register all detections
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_confidences[i])
        else:
            # Match existing objects with new detections
            object_centroids = [obj.centroid for obj in self.objects.values()]
            D = self._compute_distance_matrix(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = list(self.objects.keys())[row]
                self.objects[object_id].centroid = input_centroids[col]
                self.objects[object_id].bbox = input_bboxes[col]
                self.objects[object_id].confidence = input_confidences[col]
                self.objects[object_id].last_seen = time.time()
                self.objects[object_id].track_history.append(input_centroids[col])
                
                # Keep only recent track history
                if len(self.objects[object_id].track_history) > 10:
                    self.objects[object_id].track_history = self.objects[object_id].track_history[-10:]
                
                self.disappeared[object_id] = 0
                used_row_indices.add(row)
                used_col_indices.add(col)
                
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            # Register new detections
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = list(self.objects.keys())[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col], input_bboxes[col], input_confidences[col])
                    
        return self.objects
        
    def _compute_distance_matrix(self, object_centroids, input_centroids):
        """Compute distance matrix between object and input centroids."""
        D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids), axis=2)
        return D

class AdvancedVehicleDetector:
    """Advanced vehicle detection system with GPU acceleration and tracking."""
    
    def __init__(self, config_file=None):
        self.config = self._load_config(config_file)
        self.car_cascade = cv2.CascadeClassifier(self.config['cascade_file'])
        self.tracker = VehicleTracker(
            max_disappeared=self.config['max_disappeared'],
            max_distance=self.config['max_distance']
        )
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        self.detection_history = deque(maxlen=300)  # 10 seconds at 30 FPS
        
        # GPU acceleration setup
        self.setup_gpu_acceleration()
        
        # Statistics
        self.stats = TrafficStats()
        self.frame_count = 0
        self.start_time = time.time()
        
    def _load_config(self, config_file):
        """Load configuration from file or use defaults."""
        default_config = {
            'cascade_file': 'cars.xml',
            'camera_index': 1,
            'resize_width': 640,
            'scale_factor': 1.1,
            'min_neighbors': 3,
            'min_size': (30, 30),
            'max_disappeared': 30,
            'max_distance': 50,
            'confidence_threshold': 0.3,
            'enable_gpu': True,
            'enable_tracking': True,
            'enable_analytics': True,
            'save_video': False,
            'output_file': 'traffic_output.mp4'
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
        
    def setup_gpu_acceleration(self):
        """Setup GPU acceleration if available."""
        if self.config['enable_gpu'] and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            try:
                # Create GPU matrices
                self.gpu_frame = cv2.cuda_GpuMat()
                self.gpu_gray = cv2.cuda_GpuMat()
                self.gpu_objects = cv2.cuda_GpuMat()
                
                # Create cascade classifier for GPU
                self.gpu_cascade = cv2.cuda.CascadeClassifier_create(self.config['cascade_file'])
                self.gpu_enabled = True
                print("✅ GPU acceleration enabled")
            except Exception as e:
                print(f"⚠️ GPU setup failed, falling back to CPU: {e}")
                self.gpu_enabled = False
        else:
            self.gpu_enabled = False
            print("ℹ️ Using CPU processing")
            
    def detect_vehicles_gpu(self, frame):
        """Detect vehicles using GPU acceleration."""
        if not self.gpu_enabled:
            return self.detect_vehicles_cpu(frame)
            
        try:
            # Upload frame to GPU
            self.gpu_frame.upload(frame)
            
            # Convert to grayscale on GPU
            cv2.cuda.cvtColor(self.gpu_frame, cv2.COLOR_BGR2GRAY, self.gpu_gray)
            
            # Detect objects on GPU
            detections = self.gpu_cascade.detectMultiScale(
                self.gpu_gray,
                scaleFactor=self.config['scale_factor'],
                minNeighbors=self.config['min_neighbors'],
                minSize=self.config['min_size']
            )
            
            if len(detections) > 0:
                detections = detections[0]  # GPU returns tuple
                
            return [(x, y, w, h, 1.0) for (x, y, w, h) in detections]
            
        except Exception as e:
            print(f"GPU detection failed, falling back to CPU: {e}")
            return self.detect_vehicles_cpu(frame)
            
    def detect_vehicles_cpu(self, frame):
        """Detect vehicles using CPU processing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        cars = self.car_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config['scale_factor'],
            minNeighbors=self.config['min_neighbors'],
            minSize=self.config['min_size']
        )
        
        # Filter detections by confidence (using area as proxy)
        filtered_cars = []
        for (x, y, w, h) in cars:
            area = w * h
            confidence = min(1.0, area / (frame.shape[0] * frame.shape[1] * 0.01))
            if confidence >= self.config['confidence_threshold']:
                filtered_cars.append((x, y, w, h, confidence))
                
        return filtered_cars
        
    def update_tracking(self, detections):
        """Update vehicle tracking."""
        if not self.config['enable_tracking']:
            return {}
            
        return self.tracker.update(detections)
        
    def draw_visualization(self, frame, tracked_objects, detections):
        """Draw advanced visualization with tracking and analytics."""
        height, width = frame.shape[:2]
        
        # Draw detection rectangles
        for (x, y, w, h, conf) in detections:
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{conf:.2f}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw tracked objects with trails
        for obj in tracked_objects.values():
            x, y, w, h = obj.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Draw ID and confidence
            cv2.putText(frame, f"ID:{obj.id}", (x, y - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"{obj.confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Draw track history
            if len(obj.track_history) > 1:
                points = np.array(obj.track_history, dtype=np.int32)
                cv2.polylines(frame, [points], False, (255, 0, 0), 2)
                
        # Draw analytics overlay
        self.draw_analytics_overlay(frame, tracked_objects)
        
        return frame
        
    def draw_analytics_overlay(self, frame, tracked_objects):
        """Draw analytics overlay with traffic statistics."""
        if not self.config['enable_analytics']:
            return
            
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Traffic statistics
        current_vehicles = len(tracked_objects)
        total_vehicles = self.stats.total_vehicles
        fps = self.stats.fps
        processing_time = self.stats.processing_time
        
        # Calculate vehicles per minute
        elapsed_time = time.time() - self.start_time
        vehicles_per_minute = (total_vehicles / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        # Display statistics
        y_offset = 30
        cv2.putText(frame, f"LIVE VEHICLE DETECTION", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(frame, f"Current Vehicles: {current_vehicles}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20
        
        cv2.putText(frame, f"Total Detected: {total_vehicles}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20
        
        cv2.putText(frame, f"Vehicles/min: {vehicles_per_minute:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
        
        cv2.putText(frame, f"Processing: {processing_time:.2f}ms", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
        
        # Traffic status
        if current_vehicles >= 3:
            status = "HEAVY TRAFFIC"
            color = (0, 0, 255)
        elif current_vehicles >= 2:
            status = "MODERATE TRAFFIC"
            color = (0, 165, 255)
        elif current_vehicles >= 1:
            status = "LIGHT TRAFFIC"
            color = (0, 255, 255)
        else:
            status = "NO TRAFFIC"
            color = (0, 255, 0)
            
        cv2.putText(frame, f"Status: {status}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # GPU status indicator
        gpu_status = "GPU" if self.gpu_enabled else "CPU"
        gpu_color = (0, 255, 0) if self.gpu_enabled else (255, 0, 0)
        cv2.putText(frame, f"Processing: {gpu_status}", (width - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gpu_color, 2)
        
    def update_statistics(self, tracked_objects, processing_time):
        """Update performance and traffic statistics."""
        self.frame_count += 1
        
        # Update FPS
        current_time = time.time()
        self.fps_counter.append(current_time)
        if len(self.fps_counter) > 1:
            fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
            self.stats.fps = fps
            
        # Update processing time
        self.processing_times.append(processing_time)
        self.stats.processing_time = np.mean(self.processing_times) * 1000  # Convert to ms
        
        # Update vehicle count
        current_vehicles = len(tracked_objects)
        self.detection_history.append(current_vehicles)
        
        # Count new vehicles (simple heuristic)
        if current_vehicles > 0:
            # Check for new vehicles by comparing with recent history
            recent_avg = np.mean(list(self.detection_history)[-10:]) if len(self.detection_history) >= 10 else 0
            if current_vehicles > recent_avg * 1.2:  # 20% increase indicates new vehicles
                self.stats.total_vehicles += max(0, current_vehicles - int(recent_avg))
        
        # Update average confidence
        if tracked_objects:
            confidences = [obj.confidence for obj in tracked_objects.values()]
            self.stats.avg_confidence = np.mean(confidences)
            
    def run(self):
        """Main detection loop."""
        print("🚀 Starting Advanced Vehicle Detection System...")
        print(f"📊 Configuration: {self.config}")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.config['camera_index'])
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
            
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Video writer setup
        writer = None
        if self.config['save_video']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.config['output_file'], fourcc, 30.0, (1280, 720))
            
        print("✅ Camera initialized successfully")
        print("🎯 Press 'q' to quit, 's' to save screenshot, 'r' to reset statistics")
        
        try:
            while True:
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading frame")
                    break
                    
                # Resize frame for processing
                frame = imutils.resize(frame, width=self.config['resize_width'])
                original_frame = frame.copy()
                
                # Detect vehicles
                detections = self.detect_vehicles_gpu(frame) if self.gpu_enabled else self.detect_vehicles_cpu(frame)
                
                # Update tracking
                tracked_objects = self.update_tracking(detections)
                
                # Draw visualization
                frame = self.draw_visualization(frame, tracked_objects, detections)
                
                # Update statistics
                processing_time = time.time() - start_time
                self.update_statistics(tracked_objects, processing_time)
                
                # Display frame
                cv2.imshow("Advanced Vehicle Detection System", frame)
                
                # Save video if enabled
                if writer is not None:
                    writer.write(cv2.resize(frame, (1280, 720)))
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('r'):
                    self.stats.total_vehicles = 0
                    self.start_time = time.time()
                    print("🔄 Statistics reset")
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            
        finally:
            # Cleanup
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self.print_final_statistics()
            
    def print_final_statistics(self):
        """Print final performance statistics."""
        print("\n" + "="*50)
        print("📊 FINAL PERFORMANCE STATISTICS")
        print("="*50)
        elapsed_time = time.time() - self.start_time
        print(f"🕐 Total Runtime: {elapsed_time:.2f} seconds")
        print(f"🎬 Total Frames: {self.frame_count}")
        print(f"📈 Average FPS: {self.stats.fps:.2f}")
        print(f"🚗 Total Vehicles Detected: {self.stats.total_vehicles}")
        print(f"⚡ Average Processing Time: {self.stats.processing_time:.2f}ms")
        print(f"🎯 Average Confidence: {self.stats.avg_confidence:.2f}")
        print(f"💻 Processing Mode: {'GPU' if self.gpu_enabled else 'CPU'}")
        print("="*50)

def create_config_file():
    """Create a default configuration file."""
    config = {
        "cascade_file": "cars.xml",
        "camera_index": 1,
        "resize_width": 640,
        "scale_factor": 1.1,
        "min_neighbors": 3,
        "min_size": [30, 30],
        "max_disappeared": 30,
        "max_distance": 50,
        "confidence_threshold": 0.3,
        "enable_gpu": True,
        "enable_tracking": True,
        "enable_analytics": True,
        "save_video": False,
        "output_file": "traffic_output.mp4"
    }
    
    with open('vehicle_detection_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("✅ Configuration file created: vehicle_detection_config.json")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Advanced Vehicle Detection System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--camera', type=int, default=1, help='Camera index (default: 1)')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--no-tracking', action='store_true', help='Disable vehicle tracking')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--create-config', action='store_true', help='Create default config file')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_config_file()
        return
        
    # Create detector with configuration
    detector = AdvancedVehicleDetector(args.config)
    
    # Override config with command line arguments
    if args.camera != 1:
        detector.config['camera_index'] = args.camera
    if args.gpu:
        detector.config['enable_gpu'] = True
    if args.no_tracking:
        detector.config['enable_tracking'] = False
    if args.save_video:
        detector.config['save_video'] = True
        
    # Run the detection system
    detector.run()

if __name__ == "__main__":
    main()
