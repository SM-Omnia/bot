import os

import cv2
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.cluster import KMeans

from plotting_utils import plot_all_segments, print_sums
from slic_kmeans import get_quadrant_sums_per_segment, segment_image_slic_kmeans


def plot_segments(image, mask, quadrant_sums=None):
    # Display the image and mask side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    # Show the original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    # Show the mask
    axes[1].imshow(mask, cmap="Dark2")
    axes[1].set_title("Mask")
    h, w = image.shape[:2]

    if quadrant_sums is not None:
        print_sums(axes[1], w, h, quadrant_sums)
    axes[1].axis("off")
    plt.tight_layout()
    # plt.show()
    # # Save the figure

    return fig


def segment_image_kmeans(image: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Segments an image into `k` regions using K-means clustering in Lab color space.

    Args:
        image: Input image as a NumPy array (BGR format as from OpenCV).
        k: Number of clusters (segments).

    Returns:
        mask: 2D NumPy array of the same height and width as the input image,
              where each pixel's value is its cluster label (0 to k-1).
    """
    # Convert image to Lab color space for better perceptual separation
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    h, w, _ = lab_image.shape
    flat = lab_image.reshape((-1, 3))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(flat)

    # Reshape back to original image shape
    mask = labels.reshape((h, w)).astype(np.uint8)
    return mask


if __name__ == "__main__":
    output_dir = Path("segmentation_results_kmeans")
    os.makedirs(output_dir, exist_ok=True)

    image_dir = "/home/sarosh/Desktop/release-93e94b2/bot/images"
    image_files = []

    for file in os.listdir(image_dir):
        image_files.append(os.path.join(image_dir, file))

    # Load model (vit_b = smallest official one)
    # model = get_small_model()
    #
    # predictor = SamPredictor(model)
    # Process each image

    for img_path in image_files[:-1]:
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        masks = segment_image_slic_kmeans(image, n_segments=8)

        # Calculate the sums for each quadrant
        quadrant_sums = get_quadrant_sums_per_segment(masks)

        print("Pixel sums by quadrant:")
        for quadrant, sum_val in quadrant_sums.items():
            print(f"{quadrant}: {sum_val}")

        # fig = plot_segments(image, masks, None)

        # Create a new figure with individual segment plots
        segment_fig = plot_all_segments(image, masks, quadrant_sums)
        segment_output_path = output_dir / f"{base_name}_segments_detail.png"
        segment_fig.savefig(segment_output_path)
        plt.close(segment_fig)  # Close the figure to free memory
