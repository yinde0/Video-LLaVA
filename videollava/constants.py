CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# ======================================================================================================
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<im_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
VIDEO_PLACEHOLDER = "<video-placeholder>"

# Smart Frame Sampling Configuration
SMART_SAMPLING_MAX_CANDIDATES = 256
SMART_SAMPLING_MOTION_WEIGHT = 0.75
SMART_SAMPLING_SHARP_WEIGHT = 0.25
SMART_SAMPLING_MIN_GAP_RATIO = 0.25

# Performance Optimization Presets
SMART_SAMPLING_PRESETS = {
    'fast': {
        'max_candidates': 128,
        'motion_weight': 0.8,
        'sharp_weight': 0.2,
        'min_gap_ratio': 0.3,
        'use_gpu': False
    },
    'balanced': {
        'max_candidates': 256,
        'motion_weight': 0.75,
        'sharp_weight': 0.25,
        'min_gap_ratio': 0.25,
        'use_gpu': False
    },
    'quality': {
        'max_candidates': 512,
        'motion_weight': 0.7,
        'sharp_weight': 0.3,
        'min_gap_ratio': 0.2,
        'use_gpu': True
    },
    'ultra_fast': {
        'max_candidates': 64,
        'motion_weight': 0.9,
        'sharp_weight': 0.1,
        'min_gap_ratio': 0.4,
        'use_gpu': False
    }
}

# Default preset
SMART_SAMPLING_DEFAULT_PRESET = 'balanced'
# ======================================================================================================

MAX_IMAGE_LENGTH = 16
MAX_VIDEO_LENGTH = 1  # current video datasets only have 1 video?

PAD_LENGTH = 620
