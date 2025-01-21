from PIL import Image
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

# Load content and style images
content_image = Image.open(r"C:\Users\dksjn\style_transfer_project\content.jpg")  # Replace with your content image path
style_image = Image.open(r"C:\Users\dksjn\style_transfer_project\style.jpg")      # Replace with your style image path

# Resize images to 512x512
content_image = content_image.resize((512, 512))
style_image = style_image.resize((512, 512))

# Convert images to numpy arrays
content_array = np.array(content_image) / 255.0  # Normalize to [0, 1]
style_array = np.array(style_image) / 255.0

# Add a batch dimension for TensorFlow compatibility
content_array = np.expand_dims(content_array, axis=0)
style_array = np.expand_dims(style_array, axis=0)

# Load the pre-trained TensorFlow Hub model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Perform style transfer
stylized_image = hub_model(tf.constant(content_array), tf.constant(style_array))[0]

# Convert the output tensor to an image
stylized_image = tf.squeeze(stylized_image)  # Remove batch dimension
stylized_image = tf.keras.preprocessing.image.array_to_img(stylized_image)

# Save the stylized image
stylized_image.save("stylized_image.jpg")  # Saves the output image as 'stylized_image.jpg'

print("Style transfer complete. Stylized image saved as 'stylized_image.jpg'.")
