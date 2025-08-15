
import torch
import cv2
import decord
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from videollava.constants import (
    SMART_SAMPLING_MAX_CANDIDATES,
    SMART_SAMPLING_MOTION_WEIGHT,
    SMART_SAMPLING_SHARP_WEIGHT,
    SMART_SAMPLING_MIN_GAP_RATIO
)

decord.bridge.set_bridge('torch')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def sample_effective_frames(
    video_path: str,
    T_cap: int = 32,
    max_candidates: int = 256,
    motion_weight: float = 0.75,
    sharp_weight: float = 0.25,
    min_gap_ratio: float = 0.25
):
    """
    Select T_cap effective frames from a (possibly long) video using uniform sampling
    + motion/keyframe tweaks.

    Args:
        video_path: path to video file
        T_cap: number of frames to return
        max_candidates: number of uniformly spaced candidate frames to consider
        motion_weight: weight for motion score (frame diff)
        sharp_weight: weight for sharpness score (Laplacian variance)
        min_gap_ratio: enforce a minimum temporal gap as a fraction of candidate stride
                       (helps avoid selecting near-duplicate frames). 0.25 is safe.

    Returns:
        selected_indices: list[int] absolute frame indices (sorted, unique)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # fallback: probe by reading
        total_frames = 0
        while True:
            ok, _ = cap.read()
            if not ok: break
            total_frames += 1
        cap.release()
        cap = cv2.VideoCapture(video_path)

    # If video is short, just take unique indices up to T_cap
    if total_frames <= T_cap:
        return list(range(total_frames))

    # 1) Build candidate frame indices (uniform across the whole video)
    C = min(max_candidates, total_frames)
    cand_idx = np.linspace(0, total_frames - 1, C, dtype=np.int64)

    # 2) Compute motion + sharpness scores for each candidate
    #    Motion: mean absolute diff vs previous candidate frame (grayscale)
    #    Sharpness: Laplacian variance (higher = sharper)
    prev_gray = None
    motion_scores = np.zeros(C, dtype=np.float32)
    sharp_scores = np.zeros(C, dtype=np.float32)

    for i, idx in enumerate(cand_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Sharpness
        sharp_scores[i] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Motion (diff to previous candidate)
        if prev_gray is None:
            motion_scores[i] = 0.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores[i] = float(diff.mean())
        prev_gray = gray

    cap.release()

    # Normalize scores to [0,1] (avoid NaNs if constant)
    def norm01(x):
        x = x.astype(np.float32)
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    motion_n = norm01(motion_scores)
    sharp_n  = norm01(sharp_scores)
    score = motion_weight * motion_n + sharp_weight * sharp_n

    # 3) Stratified top-pick per temporal bin to ensure coverage
    #    Split the candidate timeline into T_cap bins and choose the top-scoring
    #    candidate in each bin. This keeps frames spread over the hour.
    bins = np.linspace(0, C, T_cap + 1, dtype=int)
    chosen = []
    used = np.zeros(C, dtype=bool)

    # Enforce a small temporal diversity within the candidate grid
    # expressed in candidate steps
    min_gap = max(1, int(min_gap_ratio * (C / T_cap)))  # e.g., with C=256, T_cap=32 -> ~2

    for b in range(T_cap):
        start, end = bins[b], bins[b + 1]
        if start >= end:
            continue
        # Among unused candidates in this bin, pick the best by score
        bin_inds = np.arange(start, end, dtype=int)
        # filter out near-duplicates around already chosen indices
        mask = np.ones_like(bin_inds, dtype=bool)
        for j, ci in enumerate(bin_inds):
            if used[ci]:
                mask[j] = False
                continue
            # Check neighborhood
            left  = max(0, ci - min_gap)
            right = min(C, ci + min_gap + 1)
            if used[max(0, left):right].any():
                mask[j] = False
        candidates = bin_inds[mask]
        if candidates.size == 0:
            # fall back: allow reused neighborhood if bin is empty
            candidates = bin_inds

        best_ci = candidates[np.argmax(score[candidates])]
        chosen.append(best_ci)
        # mark a neighborhood as used to encourage diversity
        left  = max(0, best_ci - min_gap)
        right = min(C, best_ci + min_gap + 1)
        used[left:right] = True

    # Map candidate indices back to absolute frame indices
    selected_indices = sorted(set([int(cand_idx[i]) for i in chosen]))

    # If we somehow picked fewer than T_cap (pathological), top-up by global best
    if len(selected_indices) < T_cap:
        remaining = T_cap - len(selected_indices)
        # pick from all candidates not already used, by score
        all_order = np.argsort(-score)
        for ci in all_order:
            fi = int(cand_idx[ci])
            if fi not in selected_indices:
                selected_indices.append(fi)
                if len(selected_indices) == T_cap:
                    break
        selected_indices.sort()

    return selected_indices

def sample_effective_frames_gpu(
    video_path: str,
    T_cap: int = 32,
    max_candidates: int = 256,
    motion_weight: float = 0.75,
    sharp_weight: float = 0.25,
    min_gap_ratio: float = 0.25,
    use_gpu: bool = True,
    batch_size: int = 32
):
    """
    GPU-accelerated version of smart frame sampling for better performance.
    
    Args:
        video_path: path to video file
        T_cap: number of frames to return
        max_candidates: number of uniformly spaced candidate frames to consider
        motion_weight: weight for motion score (frame diff)
        sharp_weight: weight for sharpness score (Laplacian variance)
        min_gap_ratio: enforce a minimum temporal gap as a fraction of candidate stride
        use_gpu: whether to use GPU acceleration
        batch_size: batch size for GPU processing
    
    Returns:
        selected_indices: list[int] absolute frame indices (sorted, unique)
    """
    import torch
    
    # Check GPU availability
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # fallback: probe by reading
        total_frames = 0
        while True:
            ok, _ = cap.read()
            if not ok: break
            total_frames += 1
        cap.release()
        cap = cv2.VideoCapture(video_path)

    # If video is short, just take unique indices up to T_cap
    if total_frames <= T_cap:
        return list(range(total_frames))

    # 1) Build candidate frame indices (uniform across the whole video)
    C = min(max_candidates, total_frames)
    cand_idx = np.linspace(0, total_frames - 1, C, dtype=np.int64)

    # 2) Batch load frames for GPU processing
    frames_batch = []
    for i, idx in enumerate(cand_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_batch.append(frame)
        else:
            frames_batch.append(np.zeros((224, 224, 3), dtype=np.uint8))  # fallback
    
    cap.release()
    
    # Convert to torch tensors and move to device
    frames_tensor = torch.from_numpy(np.array(frames_batch)).float().to(device)
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (C, H, W, C) -> (C, C, H, W)
    
    # 3) GPU-accelerated motion and sharpness computation
    with torch.no_grad():
        # Sharpness: Laplacian variance (higher = sharper)
        if device.type == 'cuda':
            # Use GPU-optimized operations
            frames_gray = frames_tensor[:, 0:1, :, :] * 0.299 + frames_tensor[:, 1:2, :, :] * 0.587 + frames_tensor[:, 2:3, :, :] * 0.114
            
            # GPU Laplacian kernel
            laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            laplacian_output = torch.nn.functional.conv2d(frames_gray, laplacian_kernel, padding=1)
            sharp_scores = laplacian_output.var(dim=[2, 3]).cpu().numpy().flatten()
        else:
            # CPU fallback
            sharp_scores = np.array([cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for frame in frames_batch])
        
        # Motion: frame difference (GPU-accelerated)
        if device.type == 'cuda':
            # Compute differences between consecutive frames
            frame_diffs = torch.abs(frames_tensor[1:] - frames_tensor[:-1])
            motion_scores = frame_diffs.mean(dim=[1, 2, 3]).cpu().numpy()
            motion_scores = np.concatenate([[0.0], motion_scores])  # First frame has no motion
        else:
            # CPU fallback
            motion_scores = np.zeros(C, dtype=np.float32)
            for i in range(1, C):
                diff = cv2.absdiff(frames_batch[i], frames_batch[i-1])
                motion_scores[i] = diff.mean()
    
    # 4) Score computation and normalization
    def norm01(x):
        x = x.astype(np.float32)
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    motion_n = norm01(motion_scores)
    sharp_n = norm01(sharp_scores)
    score = motion_weight * motion_n + sharp_weight * sharp_n

    # 5) Stratified selection (same as before)
    bins = np.linspace(0, C, T_cap + 1, dtype=int)
    chosen = []
    used = np.zeros(C, dtype=bool)
    min_gap = max(1, int(min_gap_ratio * (C / T_cap)))

    for b in range(T_cap):
        start, end = bins[b], bins[b + 1]
        if start >= end:
            continue
        
        bin_inds = np.arange(start, end, dtype=int)
        mask = np.ones_like(bin_inds, dtype=bool)
        
        for j, ci in enumerate(bin_inds):
            if used[ci]:
                mask[j] = False
                continue
            
            left = max(0, ci - min_gap)
            right = min(C, ci + min_gap + 1)
            if used[max(0, left):right].any():
                mask[j] = False
        
        candidates = bin_inds[mask]
        if candidates.size == 0:
            candidates = bin_inds

        best_ci = candidates[np.argmax(score[candidates])]
        chosen.append(best_ci)
        
        left = max(0, best_ci - min_gap)
        right = min(C, best_ci + min_gap + 1)
        used[left:right] = True

    # Map candidate indices back to absolute frame indices
    selected_indices = sorted(set([int(cand_idx[i]) for i in chosen]))

    # Top-up if needed
    if len(selected_indices) < T_cap:
        remaining = T_cap - len(selected_indices)
        all_order = np.argsort(-score)
        for ci in all_order:
            fi = int(cand_idx[ci])
            if fi not in selected_indices:
                selected_indices.append(fi)
                if len(selected_indices) == T_cap:
                    break
        selected_indices.sort()

    return selected_indices

def sample_effective_frames_optimized(
    video_path: str,
    preset: str = 'balanced',
    T_cap: int = 32,
    **kwargs
):
    """
    Optimized smart frame sampling using presets for easy performance tuning.
    
    Args:
        video_path: path to video file
        preset: performance preset ('ultra_fast', 'fast', 'balanced', 'quality')
        T_cap: number of frames to return
        **kwargs: override preset parameters
    
    Returns:
        selected_indices: list[int] absolute frame indices (sorted, unique)
    """
    from videollava.constants import SMART_SAMPLING_PRESETS, SMART_SAMPLING_DEFAULT_PRESET
    
    if preset not in SMART_SAMPLING_PRESETS:
        print(f"Warning: preset '{preset}' not found, using '{SMART_SAMPLING_DEFAULT_PRESET}'")
        preset = SMART_SAMPLING_DEFAULT_PRESET
    
    # Get preset parameters
    params = SMART_SAMPLING_PRESETS[preset].copy()
    
    # Override with any provided kwargs
    params.update(kwargs)
    
    # Use GPU version if specified
    if params.get('use_gpu', False):
        # Filter parameters for GPU function
        gpu_params = {k: v for k, v in params.items() if k in ['max_candidates', 'motion_weight', 'sharp_weight', 'min_gap_ratio', 'use_gpu', 'batch_size']}
        return sample_effective_frames_gpu(
            video_path=video_path,
            T_cap=T_cap,
            **gpu_params
        )
    else:
        # Filter parameters for CPU function (remove GPU-specific params)
        cpu_params = {k: v for k, v in params.items() if k in ['max_candidates', 'motion_weight', 'sharp_weight', 'min_gap_ratio']}
        return sample_effective_frames(
            video_path=video_path,
            T_cap=T_cap,
            **cpu_params
        )


def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def get_video_transform(config):
    config = config.vision_config
    if config.video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif config.video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif config.video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform


def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=32,  # Increased from 8 to 32 for better video understanding
):
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        
        # Use optimized frame sampling for better performance
        selected_frame_indices = sample_effective_frames_optimized(
            video_path=video_path,
            preset='balanced',  # Can be 'ultra_fast', 'fast', 'balanced', 'quality'
            T_cap=num_frames
        )
        
        # Convert frame indices to decord format (0-based)
        frame_id_list = np.array(selected_frame_indices, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'opencv':
        # Use optimized frame sampling for better performance
        selected_frame_indices = sample_effective_frames_optimized(
            video_path=video_path,
            preset='balanced',  # Can be 'ultra_fast', 'fast', 'balanced', 'quality'
            T_cap=num_frames
        )
        
        video_data = []
        for frame_idx in selected_frame_indices:
            cv2_vr = cv2.VideoCapture(video_path)
            cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = cv2_vr.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
            cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs

class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindVideoTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_video_transform(config)
        self.image_processor = load_and_transform_video
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            image_features = [self.image_processor(image, self.transform,
                                                   video_decode_backend=self.config.vision_config.video_decode_backend,
                                                   num_frames=self.config.vision_config.num_frames) for image in images]
            image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
