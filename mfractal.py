import os
from PIL import Image, ImageDraw, ImageFilter, ImageSequence
import numpy as np
import colorsys
import random


def extract_roi(image, brightness_threshold=180, saturation_threshold=30):
    # Convert image to numpy array
    img_data = np.array(image.convert("RGB"))

    # Calculate brightness and saturation
    brightness = np.mean(img_data, axis=2)
    saturation = np.std(img_data, axis=2)

    # Create a mask for bright/grey areas (chains)
    mask = (brightness > brightness_threshold) & (saturation < saturation_threshold)

    return mask


def generate_circular_fractal(image_size, center, base_radius, iterations=7, color_range=((30, 100), (200, 255))):
    # Create an empty image for the fractal
    fractal = Image.new("RGBA", image_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(fractal)

    # Draw Mandelbrot-inspired circular energy patterns
    for i in range(iterations):
        current_radius = base_radius * (1 - i / iterations) * np.random.uniform(0.8, 1.2)
        alpha = int(255 * (1 - i / iterations))  # Fade as the radius decreases

        # Generate colors in the purple and orange range
        hue = np.random.uniform(color_range[0][0] / 360, color_range[0][1] / 360)
        sat = np.random.uniform(0.7, 1.0)
        val = np.random.uniform(0.7, 1.0)
        rgb_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, sat, val))

        draw.ellipse(
            [
                (center[0] - current_radius, center[1] - current_radius),
                (center[0] + current_radius, center[1] + current_radius),
            ],
            outline=(*rgb_color, alpha),
            width=3,
        )

    return fractal


def generate_animated_gif(image, mask, base_radius=300, max_fractals=500, frames=35, output_path="animated_fractal.gif"):
    # Create a blank canvas for frames
    frames_list = []

    # Get mask coordinates
    mask_coords = np.column_stack(np.where(mask))

    # Check if mask_coords is empty
    if len(mask_coords) == 0:
        print("No areas detected for fractals.")
        return

    # Randomly sample a subset of coordinates
    fractal_coords = random.sample(list(mask_coords), min(max_fractals, len(mask_coords)))

    # Generate frames
    for frame_idx in range(frames):
        print(f"Generating frame {frame_idx + 1}/{frames}...")

        # Create an overlay for this frame
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

        # Adjust fractal size dynamically for animation
        dynamic_radius = base_radius * (1 + 0.1 * np.sin(2 * np.pi * frame_idx / frames))

        for idx, coord in enumerate(fractal_coords):
            fractal = generate_circular_fractal(
                image.size, center=(coord[1], coord[0]), base_radius=dynamic_radius, color_range=((270, 300), (30, 60))
            )
            overlay = Image.alpha_composite(overlay, fractal)

        # Combine overlay with the original image
        blended_frame = Image.alpha_composite(image, overlay)
        frames_list.append(blended_frame.convert("RGB"))  # Convert to RGB for GIF

    # Save as an animated GIF
    frames_list[0].save(
        output_path,
        save_all=True,
        append_images=frames_list[1:],
        duration=int(7000 / frames),  # Duration per frame (7 seconds total)
        loop=0,
    )
    print(f"Animated GIF saved to {output_path}")


def process_image(input_path, output_path, brightness_threshold=180, base_radius=300, max_fractals=500, frames=35):
    # Load the image
    image = Image.open(input_path).convert("RGBA")

    # Extract the ROI mask
    mask = extract_roi(image, brightness_threshold)

    # Generate and save the animated GIF
    generate_animated_gif(image, mask, base_radius=base_radius, max_fractals=max_fractals, frames=frames, output_path=output_path)


# Example usage
input_image = "darksoultofractal/x_BN3I0154.JPG"  # Replace with the uploaded image path
output_gif = "animated_fractal.gif"

process_image(input_image, output_gif)



# Example usage
# input_image = "darksoultofractal/x_BN3I0154.JPG"  # Replace with the uploaded image path
# output_image = "output_sparks_image.png"

# process_image(input_image, output_image)
