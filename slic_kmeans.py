import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

from scipy import ndimage

REFERENCE_COLORS = {
    "sky": [255, 255, 255],
    "wall": [220, 215, 200],  # Off-white/beige with vertical lines
    "floor": [200, 195, 180],  # Stone/pebble texture (average color)
    "ice": [100, 160, 150],
}


def extract_texture_features(image: np.ndarray, ksize: int = 31) -> np.ndarray:
    """
    Extract texture features from an image using Sobel filters.

    Args:
        image: Input image as a NumPy array (BGR format as from OpenCV)
        ksize: Kernel size for the Sobel filter (must be odd)

    Returns:
        vertical_dominance: 2D array where higher values indicate vertical patterns
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Calculate vertical dominance (higher values indicate vertical patterns)
    vertical_dominance = np.abs(ndimage.sobel(gray, axis=1)) - np.abs(
        ndimage.sobel(gray, axis=0)
    )

    # Normalize to [0, 1] range
    if vertical_dominance.max() != vertical_dominance.min():
        vertical_dominance = (vertical_dominance - vertical_dominance.min()) / (
            vertical_dominance.max() - vertical_dominance.min()
        )
    else:
        vertical_dominance = np.zeros_like(vertical_dominance)

    return vertical_dominance


def segment_image_slic_kmeans(
    image: np.ndarray,
    n_segments: int = 200,
    k: int = None,
    reference_colors=None,
    use_texture: bool = True,
) -> np.ndarray:
    # Use default reference colors if none provided
    if reference_colors is None:
        reference_colors = REFERENCE_COLORS

    if k is None:
        k = len(reference_colors)

    # Convert to RGB and then Lab
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = rgb2lab(image_rgb)

    # Generate SLIC superpixels
    segments = slic(
        image_rgb,
        n_segments=n_segments,
        compactness=10,
        start_label=0,
        enforce_connectivity=False,
    )

    # Determine feature dimensionality based on use_texture flag
    feature_dim = 4 if use_texture else 3
    num_segments = segments.max() + 1
    features = np.zeros((num_segments, feature_dim))

    # Extract texture features if requested
    if use_texture:
        vertical_dominance = extract_texture_features(image, ksize=3)

    # Compute features for each superpixel
    for seg_id in range(num_segments):
        mask = segments == seg_id
        features[seg_id, :3] = image_lab[mask].mean(axis=0)  # Mean Lab color

        if use_texture:
            features[seg_id, 3] = vertical_dominance[
                mask
            ].mean()  # Mean vertical dominance

    # Manual conversion of reference colors to Lab space
    from skimage import color

    ref_colors_rgb = np.array(list(reference_colors.values())).astype(np.uint8)
    ref_colors_rgb_shaped = ref_colors_rgb.reshape(-1, 1, 3)
    ref_colors_lab = color.rgb2lab(ref_colors_rgb_shaped).reshape(-1, 3)

    # Initialize K-means - FIX: Don't initialize with ref_features directly when using texture
    if use_texture:
        # When using texture, we can't directly initialize with ref_colors
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=1, init=ref_colors_lab)

    labels = kmeans.fit_predict(features)

    # Create a mapping from cluster IDs to consistent labels
    label_mapping = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        # Find closest reference based on features
        if use_texture:
            distances = np.sqrt(np.sum((ref_colors_lab - center[:3]) ** 2, axis=1))
        else:
            distances = np.linalg.norm(ref_colors_lab - center, axis=1)

        closest_ref_idx = np.argmin(distances)
        label_mapping[i] = closest_ref_idx

    # Apply the mapping to create the final mask
    consistent_mask = np.zeros_like(segments, dtype=np.uint8)
    for seg_id in range(num_segments):
        consistent_mask[segments == seg_id] = label_mapping[labels[seg_id]]

    return consistent_mask


def get_quadrant_sums(mask_array):
    # Normalize the mask to [0,1] range
    mask_norm = mask_array.astype(float)
    if mask_norm.max() > 0:  # Avoid division by zero
        mask_norm = mask_norm / mask_norm.max()

    # Get the dimensions of the mask
    height, width = mask_norm.shape

    # Calculate the midpoints
    mid_h = height // 2
    mid_w = width // 2

    # Split the mask into four quadrants
    top_left = mask_norm[:mid_h, :mid_w]
    top_right = mask_norm[:mid_h, mid_w:]
    bottom_left = mask_norm[mid_h:, :mid_w]
    bottom_right = mask_norm[mid_h:, mid_w:]

    # Get halves in different directions
    top_half = mask_norm[:mid_h, :]
    bottom_half = mask_norm[mid_h:, :]
    left_half = mask_norm[:, :mid_w]
    right_half = mask_norm[:, mid_w:]

    # Calculate the sum of pixel values in each region
    sums = {
        "top_left": np.sum(top_left),
        "top_right": np.sum(top_right),
        "bottom_left": np.sum(bottom_left),
        "bottom_right": np.sum(bottom_right),
        "top_half": np.sum(top_half),
        "bottom_half": np.sum(bottom_half),
        "left_half": np.sum(left_half),
        "right_half": np.sum(right_half),
    }

    quadrant_sum = (
        sums["top_left"]
        + sums["top_right"]
        + sums["bottom_left"]
        + sums["bottom_right"]
    )

    if quadrant_sum > 0:  # Avoid division by zero
        normalized_sums = {
            k: v / quadrant_sum
            for k, v in sums.items()
            if k in ["top_left", "top_right", "bottom_left", "bottom_right"]
        }

        # For top_half and bottom_half, normalize by their sum
        half_sum_tb = sums["top_half"] + sums["bottom_half"]
        if half_sum_tb > 0:
            normalized_sums["top_half"] = sums["top_half"] / half_sum_tb
            normalized_sums["bottom_half"] = sums["bottom_half"] / half_sum_tb
        else:
            normalized_sums["top_half"] = 0.0
            normalized_sums["bottom_half"] = 0.0

        # For left_half and right_half, normalize by their sum
        half_sum_lr = sums["left_half"] + sums["right_half"]
        if half_sum_lr > 0:
            normalized_sums["left_half"] = sums["left_half"] / half_sum_lr
            normalized_sums["right_half"] = sums["right_half"] / half_sum_lr
        else:
            normalized_sums["left_half"] = 0.0
            normalized_sums["right_half"] = 0.0
    else:
        normalized_sums = {k: 0.0 for k in sums}

    return normalized_sums


def get_quadrant_sums_per_segment(mask_array):
    # Get the number of segments (assuming values from 0 to max_value)
    num_segments = len(REFERENCE_COLORS)

    # Get the dimensions of the mask
    height, width = mask_array.shape

    # Calculate the midpoints
    mid_h = height // 2
    mid_w = width // 2

    # Initialize results dictionary to store sums for each segment
    results = {}

    # Process each segment
    for segment_id in range(num_segments):
        # Create a binary mask for this segment
        segment_mask = (mask_array == segment_id).astype(float)

        # Split the mask into quadrants
        top_left = segment_mask[:mid_h, :mid_w]
        top_right = segment_mask[:mid_h, mid_w:]
        bottom_left = segment_mask[mid_h:, :mid_w]
        bottom_right = segment_mask[mid_h:, mid_w:]

        # Get halves in different directions
        top_half = segment_mask[:mid_h, :]
        bottom_half = segment_mask[mid_h:, :]
        left_half = segment_mask[:, :mid_w]
        right_half = segment_mask[:, mid_w:]

        # Calculate the sum of pixel values in each region
        sums = {
            "top_left": np.sum(top_left),
            "top_right": np.sum(top_right),
            "bottom_left": np.sum(bottom_left),
            "bottom_right": np.sum(bottom_right),
            "top_half": np.sum(top_half),
            "bottom_half": np.sum(bottom_half),
            "left_half": np.sum(left_half),
            "right_half": np.sum(right_half),
        }

        quadrant_sum = (
            sums["top_left"]
            + sums["top_right"]
            + sums["bottom_left"]
            + sums["bottom_right"]
        )

        if quadrant_sum > 0:  # Avoid division by zero
            normalized_sums = {
                k: v / quadrant_sum
                for k, v in sums.items()
                if k in ["top_left", "top_right", "bottom_left", "bottom_right"]
            }

            # For top_half and bottom_half, normalize by their sum
            half_sum_tb = sums["top_half"] + sums["bottom_half"]
            if half_sum_tb > 0:
                normalized_sums["top_half"] = sums["top_half"] / half_sum_tb
                normalized_sums["bottom_half"] = sums["bottom_half"] / half_sum_tb
            else:
                normalized_sums["top_half"] = 0.0
                normalized_sums["bottom_half"] = 0.0

            # For left_half and right_half, normalize by their sum
            half_sum_lr = sums["left_half"] + sums["right_half"]
            if half_sum_lr > 0:
                normalized_sums["left_half"] = sums["left_half"] / half_sum_lr
                normalized_sums["right_half"] = sums["right_half"] / half_sum_lr
            else:
                normalized_sums["left_half"] = 0.0
                normalized_sums["right_half"] = 0.0
        else:
            normalized_sums = {k: 0.0 for k in sums}

        results[segment_id] = normalized_sums

    return results
