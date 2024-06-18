# Denoising-of-image
# RIDNet (Residual Image Denoising Network)

## Overview

RIDNet (Residual Image Denoising Network) is an advanced neural network designed for image denoising. It leverages residual connections and attention mechanisms to effectively remove noise from images while preserving fine details. This README provides an overview of the architecture, training process, and evaluation metrics used in the implementation of RIDNet.

## Architecture

### 1. *Input Layer*
- *Shape:* (256, 256, 3) representing a 256x256 RGB image.

### 2. *Feature Extraction Module*
- *Conv2D Layer:* 64 filters, kernel size (3, 3), padding 'same'.

### 3. *Feature Learning Residual on Residual Module (EAM)*
- *EAM Modules:* A series of Enhanced Attention Modules (EAM) are applied to the extracted features. These modules are responsible for learning complex features and emphasizing important regions of the image.
- *Number of EAMs:* 4

### 4. *Reconstruction Module*
- *Conv2D Layer:* 3 filters, kernel size (3, 3), padding 'same'.
- *Add Layer:* A residual connection that adds the original input image to the output of the final convolutional layer to reconstruct the denoised image.

### 5. *Model Summary*
plaintext
Model: "RIDNet"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 256, 256, 3)]     0         
                                                                 
 conv2d (Conv2D)             (None, 256, 256, 64)      1792      
                                                                 
 eam (EAM)                   multiple                  73920     
                                                                 
 conv2d_1 (Conv2D)           (None, 256, 256, 3)       1731      
                                                                 
 add (Add)                   (None, 256, 256, 3)       0         
                                                                 
=================================================================
Total params: 77,443
Trainable params: 77,443
Non-trainable params: 0
_________________________________________________________________


## Training

### 1. *Data Preparation*
- *Training Data:* High-quality and corresponding low-quality images from specified directories.
- *Testing Data:* High-quality and corresponding low-quality images from specified directories.

### 2. *Data Generators*
- Custom data generators were created to load and preprocess the images efficiently during training and testing.

### 3. *Model Compilation*
- *Optimizer:* Adam with a learning rate of 1e-3.
- *Loss Function:* Mean Squared Error (MSE).

### 4. *Training Process*
python
# Example usage of DataGenerator with the RIDNet
batch_size = 32
train_generator = DataGenerator(train_high_paths, train_low_paths, batch_size=batch_size, shuffle=True)
test_generator = DataGenerator(test_high_paths, test_low_paths, batch_size=batch_size, shuffle=True)

# Define callbacks
checkpoint_callback = ModelCheckpoint(filepath='denoiser_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
history = RIDNet.fit(train_generator,
                     epochs=1,
                     validation_data=test_generator,
                     callbacks=[checkpoint_callback])


### 5. *Saving the Model*
python
# Save the trained model
RIDNet.save('denoiser_model.h5')


## Evaluation

### 1. *Evaluation Metrics*
- *PSNR (Peak Signal-to-Noise Ratio):* Measures the ratio between the maximum possible power of a signal and the power of corrupting noise.
- *SSIM (Structural Similarity Index):* Measures the similarity between two images.

### 2. *Evaluation Function*
The denoise_and_evaluate function was created to:
- Denoise images using the trained RIDNet.
- Compute average PSNR and SSIM values for the test dataset.

python
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def denoise_and_evaluate(generator, model):
    psnr_values = []
    ssim_values = []
    num_batches = len(generator)

    for i in range(num_batches):
        X_batch, y_batch = generator[i]
        denoised_batch = model.predict(X_batch)

        for j in range(len(X_batch)):
            original_image = X_batch[j]
            noisy_image = y_batch[j]
            denoised_image = denoised_batch[j]

            # Compute PSNR
            psnr = peak_signal_noise_ratio(original_image, denoised_image)
            psnr_values.append(psnr)

            # Compute SSIM
            ssim = structural_similarity(original_image, denoised_image, multichannel=True)
            ssim_values.append(ssim)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim

# Calculate PSNR and SSIM for test set
avg_psnr, avg_ssim = denoise_and_evaluate(test_generator, RIDNet)

print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")


### 3. *Results*
- *Average PSNR:* 27.45.
  

## Usage

### 1. *Inference on Test Images*
To denoise images from the ./test/low/ directory and save the results to ./test/predicted/:

python
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('denoiser_model.h5')

# Directory paths
input_dir = './test/low/'
output_dir = './test/predicted/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Denoise and save images
for filename in os.listdir(input_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Load and preprocess the image
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict denoised image
        denoised_img = model.predict(img)
        denoised_img = np.squeeze(denoised_img, axis=0)  # Remove batch dimension
        denoised_img = (denoised_img * 255).astype(np.uint8)  # Rescale to [0, 255]

        # Save the denoised image
        output_path = os.path.join(output_dir, filename)
        Image.fromarray(denoised_img).save(output_path)


## Conclusion

RIDNet offers a robust framework for image denoising by utilizing advanced techniques like residual connections and attention mechanisms. With proper training and tuning, RIDNet can effectively reduce noise in images, improving both PSNR and SSIM values, leading to clearer and more visually appealing results. This README provides a comprehensive guide to setting up, training, and evaluating RIDNet, making it easier for users to replicate and build upon this work.

---
