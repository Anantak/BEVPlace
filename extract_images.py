import cv2
import os
import cv2
import os

def extract_rectangles_from_directory(image_dir, output_dir, x, y, width, height, resize_factor=None):
  """
  Extracts rectangular portions from all images in a directory, saves them,
  and optionally resizes them.

  Args:
      image_dir: Path to the directory containing images (str, absolute or relative).
      output_dir: Directory to save the extracted images (str, absolute or relative).
      x: Top-left X coordinate of the rectangle (int).
      y: Top-left Y coordinate of the rectangle (int).
      width: Width of the rectangle (int).
      height: Height of the rectangle (int).
      resize_factor: A float representing the factor to resize the extracted image 
                      (optional, float). This will resize to `width * resize_factor` 
                      and `height * resize_factor`.

  Returns:
      None if errors occur, otherwise continues processing.
  """
  for filename in os.listdir(image_dir):
    # Get the full image path
    image_path = os.path.join(image_dir, filename)

    # Check if it's a valid image file
    if os.path.isfile(image_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
      try:
        extracted_image = extract_rectangle(image_path, x, y, width, height, output_dir)

        # Resize the extracted image (if resize_factor is provided)
        if resize_factor is not None:
          # Ensure integer resize dimensions
          resize_dim = (int(width * resize_factor), int(height * resize_factor))
          extracted_image = cv2.resize(extracted_image, resize_dim)
          print(f"Resized image to: {resize_dim}")

        # Save the resized or original extracted image
        save_extracted_image(image_path, extracted_image, output_dir)
      except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def extract_rectangle(image_path, x, y, width, height, output_dir):
  """
  Extracts a rectangular portion from an image and saves it.

  Args:
      image_path: Path to the image file (str).
      x: Top-left X coordinate of the rectangle (int).
      y: Top-left Y coordinate of the rectangle (int).
      width: Width of the rectangle (int).
      height: Height of the rectangle (int).
      output_dir: Directory to save the extracted image (str).

  Returns:
      The extracted image as a NumPy array, or None if errors occur.
  """
  try:
    # Read the image
    image = cv2.imread(image_path)

    # Check if image read successfully
    if image is None:
      print(f"Error: Could not read image from {image_path}")
      return None

    # Extract the rectangular portion
    extracted_image = image[y:y+height, x:x+width]

    return extracted_image
  except Exception as e:
    print(f"Error extracting rectangle from {image_path}: {e}")
    return None

def save_extracted_image(image_path, extracted_image, output_dir):
  """
  Saves the extracted image with the same filename and extension.

  Args:
      image_path: Path to the original image file (str).
      extracted_image: The extracted image as a NumPy array.
      output_dir: Directory to save the extracted image (str).
  """
  # Create output filename with same extension as original image
  filename, extension = os.path.splitext(os.path.basename(image_path))
  # print(filename, extension)
  output_filename = f"{filename}_extracted{extension}"

  # Create output directory if it doesn't exist
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Save the extracted image
  output_path = os.path.join(output_dir, output_filename)
  cv2.imwrite(output_path, extracted_image)
  print(f"Extracted rectangle saved to: {output_path}")

# Define image directory, rectangle coordinates, output directory

data_path = '/home/ubuntu/Downloads/BEV'
seq = '08'

# Define image directory, rectangle coordinates, and output directory
image_dir = f"/home/ubuntu/Anantak/SensorUnit/data/Map/ImageData/0{seq}/05"
output_dir = f"{data_path}/{seq}/images"
width = 550
height = 550
x = int(848/2-width/2)
y = int(848/2-height/2)
resize_factor = 151/width

# Call the main function
extract_rectangles_from_directory(image_dir, output_dir, x, y, width, height, resize_factor)
