import sys
import os
from PIL import Image

# 1. Check if enough arguments were provided
if len(sys.argv) < 3:
    print("Error: Missing arguments.")
    print("Usage: python3 script.py <file_path> <top_crop_pixels>")
    sys.exit(1)

# 2. Assign arguments
file_path = sys.argv[1]
try:
    top_crop_value = int(sys.argv[2])
except ValueError:
    print("Error: The top crop value must be an integer.")
    sys.exit(1)

# 3. Open image and calculate crop
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.")
    sys.exit(1)

img = Image.open(file_path)
width, height = img.size

# Set coordinates
left = 0
top = top_crop_value
right = width
bottom = height

# Execute crop
cropped_img = img.crop((left, top, right, bottom))

# 4. Generate the save name 
# This follows your pattern: [original_name]processed_screenshot.png
save_name = file_path + "processed_screenshot.png"
cropped_img.save(save_name)

# 5. Output results
print(f"File processed: {file_path}")
print(f"Original size:  {width}x{height}")
print(f"New size:       {cropped_img.size[0]}x{cropped_img.size[1]}")
print(f"Saved as:       {save_name}")
