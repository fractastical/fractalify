import os
from PIL import Image, ImageDraw
import numpy as np
import colorsys
import random


def extract_dark_roi(image, brightness_threshold=80, saturation_threshold=30):
    # Convert image to numpy array
    img_data = np.array(image.convert("RGB"))

    # Calculate brightness and saturation
    brightness = np.mean(img_data, axis=2)
    saturation = np.std(img_data, axis=2)

    # Create a mask for dark areas
    mask = (brightness < brightness_threshold) & (saturation < saturation_threshold)

    return mask


def generate_evolving_fractal(image_size, center, sizes, frame_index, total_frames, color_range):
    # Create an empty image for the fractal
    fractal = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(fractal)

    # Generate circles of 5 different sizes
    for size_idx, base_radius in enumerate(sizes):
        # Calculate evolving radius for each size
        evolving_radius = base_radius * (1 + 0.1 * np.sin(2 * np.pi * frame_index / total_frames + size_idx))

        # Generate colors in the blue-purple range
        hue = np.random.uniform(220 / 360, 280 / 360)  # Blue to purple range
        sat = np.random.uniform(0.5, 0.7)  # Slightly muted saturation
        val = np.random.uniform(0.8, 1.0)  # Lighter shades
        rgb_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, sat, val))
        alpha = int(255 * (1 - size_idx / len(sizes)))  # Fade smaller circles more

        # Draw circle
        draw.ellipse(
            [
                (center[0] - evolving_radius, center[1] - evolving_radius),
                (center[0] + evolving_radius, center[1] + evolving_radius),
            ],
            outline=(*rgb_color, alpha),
            width=3,
        )

    return fractal


def generate_smooth_dark_animated_gif(image, mask, base_sizes, max_fractals=100, frames=35, output_path="smooth_dark_evolving_fractal.gif"):
    # Create a blank canvas for frames
    frames_list = []

    # Get mask coordinates
    mask_coords = np.column_stack(np.where(mask))

    # Check if mask_coords is empty
    if len(mask_coords) == 0:
        print("No dark areas detected for fractals.")
        return

    # Randomly sample a subset of coordinates
    fractal_coords = random.sample(list(mask_coords), min(max_fractals, len(mask_coords)))

    # Generate evolving fractals for each frame
    for frame_idx in range(frames):
        print(f"Generating frame {frame_idx + 1}/{frames}...")

        # Create an overlay for this frame
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

        for idx, coord in enumerate(fractal_coords):
            # Ensure the center stays within the darkness mask
            if mask[coord[0], coord[1]]:
                fractal = generate_evolving_fractal(
                    image.size,
                    center=(coord[1], coord[0]),
                    sizes=base_sizes,
                    frame_index=frame_idx,
                    total_frames=frames,
                    color_range=((220, 280), (0.8, 1.0)),
                )
                overlay = Image.alpha_composite(overlay, fractal)

        # Combine overlay with the original image
        blended_frame = Image.alpha_composite(image, overlay)
        frames_list.append(blended_frame.convert("RGB"))  # Convert to RGB for GIF

    # Save as an animated GIF
    print("Saving GIF...")
    frames_list[0].save(
        output_path,
        save_all=True,
        append_images=frames_list[1:],
        duration=int(7000 / frames),  # Duration per frame (7 seconds total)
        loop=0,
        optimize=False,  # Disable optimization for speed
    )
    print(f"Animated GIF saved to {output_path}")


def process_dark_constrained_image(input_path, output_path, brightness_threshold=80, base_sizes=(50, 100, 150, 200, 250), max_fractals=100, frames=35):
    # Load the image
    image = Image.open(input_path).convert("RGBA")

    # Extract the dark ROI mask
    mask = extract_dark_roi(image, brightness_threshold)

    # Generate and save the animated GIF
    generate_smooth_dark_animated_gif(image, mask, base_sizes=base_sizes, max_fractals=max_fractals, frames=frames, output_path=output_path)


# Example usage
input_image = "darksoultofractal/BN3I0447.JPG"  # Replace with the uploaded image path
output_gif = "animated_fractal5.gif"

process_dark_constrained_image(input_image, output_gif)
