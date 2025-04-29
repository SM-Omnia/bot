import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_single_segment(axis, image, mask, segment_id, sums):
    """
    Plot a single segment mask on the given axis and annotate with quadrant sums.

    Args:
        axis: Matplotlib axis to plot on
        image: Original image
        mask: Segmentation mask
        segment_id: ID of the segment to plot
        sums: Dictionary with normalized distribution metrics for this segment
    """
    # Create a binary mask for this segment
    segment_mask = (mask == segment_id).astype(np.uint8)

    # Create a colored overlay for visualization
    colored_mask = np.zeros_like(image)
    colored_mask[segment_mask == 1] = [255, 0, 0]  # Red for the segment

    # Blend with original image for context
    alpha = 0.8
    blend = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    # Display the blended image
    axis.imshow(blend)
    axis.set_title(f"Segment {segment_id}")

    # Add quadrant sums
    h, w = image.shape[:2]
    print_sums(axis, w, h, sums)


def plot_all_segments(image, mask, quadrant_sums_per_segment):
    """
    Create a figure with subplots for each segment in the mask.

    Args:
        image: Original image
        mask: Segmentation mask
        quadrant_sums_per_segment: Dictionary with normalized distribution metrics for each segment

    Returns:
        fig: Matplotlib figure with all segment subplots
    """
    # Get number of segments in the dictionary
    num_segments = len(quadrant_sums_per_segment)

    if num_segments == 0:
        # If no segments, create a simple figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        ax.imshow(image)
        ax.set_title("No segments found")
        ax.axis("off")
        return fig

    # Calculate grid dimensions (trying to make it approximately square)
    import math
    grid_size = math.ceil(math.sqrt(num_segments))
    rows = math.ceil(num_segments / grid_size)
    cols = grid_size

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))

    # Convert to flattened array for easier indexing if there's only one row or column
    if num_segments == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each segment
    for i, (segment_id, sums) in enumerate(quadrant_sums_per_segment.items()):
        if i < len(axes):
            plot_single_segment(axes[i], image, mask, segment_id, sums)

    # Hide any unused subplots
    for i in range(num_segments, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()


    return fig


def print_sums(axis, width, height, sums):
    # Add quadrant sums as text annotations
    # Format the values to 2 decimal places
    axis.text(
        width * 0.25,
        height * 0.25,
        f"TL: {sums['top_left']:.2f}",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.7),
    )

    axis.text(
        width * 0.75,
        height * 0.25,
        f"TR: {sums['top_right']:.2f}",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.7),
    )

    axis.text(
        width * 0.25,
        height * 0.75,
        f"BL: {sums['bottom_left']:.2f}",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.7),
    )

    axis.text(
        width * 0.75,
        height * 0.75,
        f"BR: {sums['bottom_right']:.2f}",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.7),
    )

    # Add top and bottom half sums
    axis.text(
        width * 0.5,
        height * 0.1,
        f"Top: {sums['top_half']:.2f}",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        bbox=dict(facecolor="blue", alpha=0.7),
    )

    axis.text(
        width * 0.5,
        height * 0.9,
        f"Bottom: {sums['bottom_half']:.2f}",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        bbox=dict(facecolor="blue", alpha=0.7),
    )

    # Add left and right half sums
    axis.text(
        width * 0.1,
        height * 0.5,
        f"Left: {sums['left_half']:.2f}",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        bbox=dict(facecolor="green", alpha=0.7),
    )

    axis.text(
        width * 0.9,
        height * 0.5,
        f"Right: {sums['right_half']:.2f}",
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        bbox=dict(facecolor="green", alpha=0.7),
    )

    axis.axis("off")
