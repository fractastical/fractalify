import numpy as np
from PIL import Image, ImageDraw
import colorsys


def extract_black_regions(image, brightness_threshold=30):
    # Convert the image to numpy array
    img_data = np.array(image.convert("L"))  # Convert to grayscale

    # Create a mask for completely black regions
    mask = img_data < brightness_threshold

    return mask


def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter


def generate_mandelbrot_frame(image_size, mask, frame_index, total_frames, zoom_factor, color_scheme, max_iter=50):
    # Create an empty image for the fractal
    fractal = Image.new("RGBA", image_size, (0, 0, 0, 0))
    width, height = image_size
    draw = ImageDraw.Draw(fractal)

    # Evolve zoom and translation based on frame index
    zoom = zoom_factor ** frame_index
    x_center, y_center = -0.5, 0  # Mandelbrot set center
    x_min = x_center - 1.5 / zoom
    x_max = x_center + 1.5 / zoom
    y_min = y_center - 1 / zoom
    y_max = y_center + 1 / zoom
    x_scale = (x_max - x_min) / width
    y_scale = (y_max - y_min) / height

    # Generate the Mandelbrot fractal within the mask
    for y in range(height):
        for x in range(width):
            if mask[y, x]:  # Only draw in the completely black regions
                c = complex(x * x_scale + x_min, y * y_scale + y_min)
                m = mandelbrot(c, max_iter)
                color = color_scheme(m, max_iter, frame_index, total_frames)
                draw.point((x, y), fill=color)

    return fractal


def evolving_color_scheme(iteration, max_iter, frame_index, total_frames):
    if iteration == max_iter:
        return (0, 0, 0, 0)  # Transparent for points in the set
    hue = 220 + (iteration / max_iter) * 60 + (frame_index / total_frames) * 60  # Smooth hue shift
    sat = 0.7
    val = 1.0
    r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360, sat, val)]
    return (r, g, b, 255)  # Opaque


def generate_mandelbrot_animation(image, mask, total_frames, zoom_factor, color_scheme, max_iter, output_path):
    frames_list = []

    for frame_index in range(total_frames):
        print(f"Generating frame {frame_index + 1}/{total_frames}...")

        # Generate Mandelbrot fractal for the current frame
        fractal_frame = generate_mandelbrot_frame(
            image.size, mask, frame_index, total_frames, zoom_factor, color_scheme, max_iter
        )

        # Overlay fractal onto the original image
        blended_frame = Image.alpha_composite(image.convert("RGBA"), fractal_frame)
        frames_list.append(blended_frame.convert("RGB"))  # Convert to RGB for GIF

    # Save as an animated GIF
    print("Saving GIF...")
    frames_list[0].save(
        output_path,
        save_all=True,
        append_images=frames_list[1:],
        duration=int(7000 / total_frames),  # Duration per frame (7 seconds total)
        loop=0,
        optimize=False,  # Disable optimization for speed
    )
    print(f"Animated GIF saved to {output_path}")


def process_evolving_mandelbrot(input_path, output_path, brightness_threshold=30, total_frames=35, zoom_factor=1.1, max_iter=50):
    # Load the input image
    image = Image.open(input_path).convert("RGBA")

    # Extract black regions
    black_mask = extract_black_regions(image, brightness_threshold)

    # Generate and save the animated Mandelbrot GIF
    generate_mandelbrot_animation(
        image, black_mask, total_frames, zoom_factor, evolving_color_scheme, max_iter, output_path
    )


# Example usage
input_image = "path_to_your_image.png"  # Replace with your input image path
output_gif = "evolving_mandelbrot.gif"

process_evolving_mandelbrot(input_image, output_gif)
