import os

# Set your paths
image_dir = 'data/negative_samples'
label_dir = 'data/negative_labels'

# Create label directory if it doesn't exist
os.makedirs(label_dir, exist_ok=True)

# List of image extensions to support
image_extensions = ['.jpg', '.jpeg', '.png']

# Loop over all image files
for filename in os.listdir(image_dir):
    name, ext = os.path.splitext(filename)
    if ext.lower() in image_extensions:
        label_path = os.path.join(label_dir, f'{name}.txt')
        if not os.path.exists(label_path):
            # Create an empty .txt file
            open(label_path, 'w').close()
            print(f'Created empty label for: {name}')
