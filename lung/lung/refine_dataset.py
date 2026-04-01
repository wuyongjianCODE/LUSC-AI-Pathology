import os
import numpy as np
import cv2
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse


def main(args):
    # Parameters
    DISTANCE_THRESHOLD = args.distance  # in micrometers (um)
    PIXEL_TO_UM = np.sqrt(0.017 * 1000000)
    DISTANCE_THRESHOLD_PIXELS = int(DISTANCE_THRESHOLD / PIXEL_TO_UM)  # Convert distance to pixels

    # Create output directory
    output_dir = f"selected_patches_distance_{DISTANCE_THRESHOLD}"
    os.makedirs(output_dir, exist_ok=True)

    # Process each saved classification image
    for img_name in os.listdir(args.input_dir):
        if img_name.endswith(".png") and "_whole_classify_" in img_name:
            # Extract SVS name
            svs_name = img_name.split("_whole_classify_")[0]
            print(f"Processing {svs_name}")

            # Load the classification image
            img_path = os.path.join(args.input_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            # Convert to RGB (OpenCV loads as BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Create a mask for blue (tumor) regions
            lower_blue = np.array([0, 0, 200])
            upper_blue = np.array([100, 100, 255])
            blue_mask = cv2.inRange(img, lower_blue, upper_blue)

            # Check if there are any blue (tumor) regions
            if np.sum(blue_mask) > 0:
                # Find blue (tumor) connected components
                labeled_blue = measure.label(blue_mask)
                blue_regions = measure.regionprops(labeled_blue)

                if len(blue_regions) > 0:
                    largest_blue_region = max(blue_regions, key=lambda region: region.area)
                    blue_coords = np.array(largest_blue_region.coords)

                    # Create a mask for non-tumor regions (excluding white background)
                    lower_non_tumor = np.array([100, 100, 0])
                    upper_non_tumor = np.array([255, 255, 200])
                    non_tumor_mask = cv2.inRange(img, lower_non_tumor, upper_non_tumor)

                    # Find coordinates of non-tumor regions
                    non_tumor_coords = np.column_stack(np.where(non_tumor_mask > 0))

                    # Calculate distance from each non-tumor patch to the blue region
                    selected_patches = []
                    for coord in non_tumor_coords:
                        # Calculate minimum distance to any blue region coordinate
                        distances = np.sqrt(np.sum((blue_coords - coord) ** 2, axis=1))
                        min_distance = np.min(distances)

                        # Check if the distance is within the threshold
                        if min_distance <= DISTANCE_THRESHOLD_PIXELS:
                            # Convert coordinates to patch indices (i, j)
                            j = coord[1]
                            i = coord[0]
                            selected_patches.append((i, j))

                    # Save selected patches if any
                    if selected_patches:
                        save_selected_patches(selected_patches, svs_name, output_dir)
                    continue

            # If no blue (tumor) regions, check for red regions
            lower_red = np.array([200, 0, 0])
            upper_red = np.array([255, 100, 100])
            red_mask = cv2.inRange(img, lower_red, upper_red)

            if np.sum(red_mask) > 0:
                # Find red connected components
                labeled_red = measure.label(red_mask)
                red_regions = measure.regionprops(labeled_red)
                red_region = max(red_regions, key=lambda region: region.area)
                if len(red_regions) > 0:
                    # Select the first red region
                    first_red_region = red_region
                    red_centroid = first_red_region.centroid

                    # Create a mask for all non-background regions
                    lower_non_bg = np.array([10, 10, 10])
                    upper_non_bg = np.array([255, 255, 255])
                    non_bg_mask = cv2.inRange(img, lower_non_bg, upper_non_bg)

                    # Find coordinates of all non-background regions
                    non_bg_coords = np.column_stack(np.where(non_bg_mask > 0))

                    # Calculate distance from each patch to the red centroid
                    selected_patches = []
                    for coord in non_bg_coords:
                        distance = np.sqrt(np.sum((coord - red_centroid) ** 2))

                        # Check if the distance is within the threshold
                        if distance <= DISTANCE_THRESHOLD_PIXELS:
                            # Convert coordinates to patch indices (i, j)
                            j = coord[1]
                            i = coord[0]
                            selected_patches.append((i, j))

                    # Save selected patches if any
                    if selected_patches:
                        save_selected_patches(selected_patches, svs_name, output_dir)
                    continue

            # If no blue or red regions, use the center of the image
            img_height, img_width = img.shape[:2]
            center_x, center_y = img_width // 2, img_height // 2

            # Create a mask for all non-background regions
            lower_non_bg = np.array([10, 10, 10])
            upper_non_bg = np.array([255, 255, 255])
            non_bg_mask = cv2.inRange(img, lower_non_bg, upper_non_bg)

            # Find coordinates of all non-background regions
            non_bg_coords = np.column_stack(np.where(non_bg_mask > 0))

            # Calculate distance from each patch to the image center
            selected_patches = []
            for coord in non_bg_coords:
                distance = np.sqrt((coord[1] - center_x) ** 2 + (coord[0] - center_y) ** 2)

                # Check if the distance is within the threshold
                if distance <= DISTANCE_THRESHOLD_PIXELS:
                    # Convert coordinates to patch indices (i, j)
                    j = coord[1]
                    i = coord[0]
                    selected_patches.append((i, j))

            # Save selected patches if any
            if selected_patches:
                save_selected_patches(selected_patches, svs_name, output_dir)


def save_selected_patches(patches, svs_name, output_dir):
    """Save the list of selected patches to a text file."""
    output_file = os.path.join(output_dir, f"{svs_name}_selected_patches.txt")
    with open(output_file, 'w') as f:
        for i, j in patches:
            # Convert to 1-based indexing for patch naming
            x, y = j + 1, i + 1
            f.write(f"{svs_name}_x{x}_y{y}\n")
    print(f"Saved {len(patches)} selected patches to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select patches based on distance to tumor or other regions')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing classification images')
    parser.add_argument('--distance', type=int, default=500, help='Distance threshold in micrometers (default: 500um)')
    parser.add_argument('--plt', action='store_true', help='Show visualizations')
    args = parser.parse_args()

    main(args)