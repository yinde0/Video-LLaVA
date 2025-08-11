#!/usr/bin/env python3
"""
Test script for the new smart frame sampling functionality in Video-LLaVA.
This script tests the sample_effective_frames function with a sample video.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'videollava'))

from videollava.model.multimodal_encoder.languagebind.video.processing_video import sample_effective_frames

def test_smart_sampling():
    """Test the smart frame sampling function."""
    
    # Test parameters
    test_video_path = "test_video.mp4"  # Replace with actual test video path
    T_cap = 16  # Test with 16 frames
    
    print("Testing Smart Frame Sampling...")
    print(f"Video path: {test_video_path}")
    print(f"Target frames: {T_cap}")
    
    try:
        # Test the smart sampling function
        selected_frames = sample_effective_frames(
            video_path=test_video_path,
            T_cap=T_cap,
            max_candidates=128,  # Use smaller candidate pool for testing
            motion_weight=0.75,
            sharp_weight=0.25,
            min_gap_ratio=0.25
        )
        
        print(f"✅ Successfully selected {len(selected_frames)} frames")
        print(f"Selected frame indices: {selected_frames}")
        
        # Verify we got the expected number of frames
        if len(selected_frames) == T_cap:
            print("✅ Frame count matches expected value")
        else:
            print(f"⚠️  Frame count mismatch: expected {T_cap}, got {len(selected_frames)}")
        
        # Verify frames are sorted and unique
        if selected_frames == sorted(set(selected_frames)):
            print("✅ Frames are sorted and unique")
        else:
            print("⚠️  Frames are not properly sorted or contain duplicates")
            
    except FileNotFoundError:
        print(f"❌ Test video not found: {test_video_path}")
        print("Please provide a valid test video path")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    test_smart_sampling() 