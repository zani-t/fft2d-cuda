# create_animation.py

import os
import imageio
import matplotlib.pyplot as plt
import cv2

def create_animation(image_dir, output_filename, fps=10):
    """
    Creates an animation (GIF) from pairs of PNG frames in the specified directory.
    Each pair consists of an individual channel image and the cumulative reconstruction image.

    :param image_dir: Directory containing PNG frames.
    :param output_filename: Output GIF filename.
    :param fps: Frames per second for the animation.
    """
    # Get list of individual channel and cumulative images sorted numerically
    channel_files = []
    cumulative_files = []
    
    for f in os.listdir(image_dir):
        if f.endswith('_channel.png'):
            try:
                step = int(f.split('_')[1])
                channel_files.append((step, f))
            except ValueError:
                print(f"Warning: Skipping file with unexpected format: {f}")
        elif f.endswith('_cumulative.png'):
            try:
                step = int(f.split('_')[1])
                cumulative_files.append((step, f))
            except ValueError:
                print(f"Warning: Skipping file with unexpected format: {f}")
    
    # Sort the files based on step number
    channel_files.sort(key=lambda x: x[0])
    cumulative_files.sort(key=lambda x: x[0])
    
    # Ensure both lists have the same steps
    steps = [step for step, _ in channel_files]
    cumulative_steps = [step for step, _ in cumulative_files]
    
    common_steps = sorted(set(steps).intersection(set(cumulative_steps)))
    
    if not common_steps:
        print("Error: No matching channel and cumulative frames found.")
        return
    
    frames = []
    
    for step in common_steps:
        # Find channel file
        channel_filename = next((f for s, f in channel_files if s == step), None)
        cumulative_filename = next((f for s, f in cumulative_files if s == step), None)
        
        if channel_filename and cumulative_filename:
            channel_path = os.path.join(image_dir, channel_filename)
            cumulative_path = os.path.join(image_dir, cumulative_filename)
            
            # Read images
            channel_img = cv2.imread(channel_path)
            cumulative_img = cv2.imread(cumulative_path)
            
            if channel_img is None:
                print(f"Warning: Could not read channel image {channel_filename}")
                continue
            if cumulative_img is None:
                print(f"Warning: Could not read cumulative image {cumulative_filename}")
                continue
            
            # Resize images if necessary to ensure they are the same size
            if channel_img.shape != cumulative_img.shape:
                cumulative_img = cv2.resize(cumulative_img, (channel_img.shape[1], channel_img.shape[0]))
            
            # Combine images side by side
            combined_img = cv2.hconcat([channel_img, cumulative_img])
            
            # Convert BGR to RGB for proper color display in GIF
            combined_img_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
            
            frames.append(combined_img_rgb)
        else:
            print(f"Warning: Missing pair for step {step}. Channel: {channel_filename}, Cumulative: {cumulative_filename}")
    
    if not frames:
        print("Error: No frames to create animation.")
        return
    
    try:
        # Create and save the GIF
        imageio.mimsave(output_filename, frames, fps=fps)
        print(f"Animation saved as {output_filename}")
    except Exception as e:
        print(f"Error: Failed to create animation. {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create animation from FFT reconstruction frames.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing PNG frames.')
    parser.add_argument('--output', type=str, default='animation.gif', help='Output GIF filename.')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the GIF.')

    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"Error: The specified image directory does not exist: {args.image_dir}")
        exit(1)

    create_animation(args.image_dir, args.output, args.fps)
