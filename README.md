# Breast Cancer Detection Using Convolutional Neural Networks (CNN)

## Overview

This project is a deep learning model designed to classify breast ultrasound images into three categories: **Benign**, **Malignant**, and **Normal**. The model is built using TensorFlow/Keras and employs Convolutional Neural Networks (CNN) to analyze and classify images with high accuracy.

The final model achieved an accuracy of **92.92%** on the test dataset, with detailed performance metrics available in the classification report and confusion matrix.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Model Architecture](#model-architecture)
3. [Data Preprocessing](#data-preprocessing)
4. [Training the Model](#training-the-model)
5. [Evaluation](#evaluation)
6. [Example Usage](#example-usage)
7. [Requirements](#requirements)
8. [Usage](#usage)
9. [Results](#results)
10. [Conclusion](#conclusion)
11. [References](#references)

## Project Structure

- **Train/**: Directory containing the training images, organized by class (`Benign`, `Malignant`, `Normal`).
- **Valid/**: Directory containing the validation images, organized by class.
- **Test/**: Directory containing the test images, organized by class.
- **best_model.h5**: The saved model with the best validation accuracy.
- **README.md**: This file, explaining the project.

## Model Architecture

The CNN model consists of the following layers:

- **Convolutional Layers**: Four convolutional layers with increasing filter sizes (64, 128, 256, 512) and ReLU activation.
- **Pooling Layers**: MaxPooling layers following each convolutional layer to downsample the feature maps.
- **Flatten Layer**: Flattening the pooled feature maps into a 1D vector.
- **Fully Connected Layers**: Two dense layers with 256 and 128 units, respectively, each followed by a Dropout layer for regularization.
- **Output Layer**: A softmax activation layer with three units for multi-class classification.

## Data Preprocessing

Data preprocessing includes:

- **Rescaling**: All images are rescaled to the range [0, 1].
- **Augmentation**: The training images are augmented with random shearing, zooming, and horizontal flipping to increase the diversity of the training data.

The data generators are configured as follows:

- **Training Set**: Images are augmented and rescaled.
- **Validation/Test Set**: Images are only rescaled.

## Training the Model

The model is trained using the Adam optimizer and categorical cross-entropy loss. Early stopping and model checkpointing are employed to prevent overfitting and to save the best-performing model based on validation accuracy.

```python
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
```

## Evaluation

The model is evaluated on the test dataset, achieving the following metrics:

- **Test Accuracy**: 92.92%
- **Classification Report**: Precision, recall, and F1-score for each class.
- **Confusion Matrix**: Visual representation of the model's performance on each class.
![image](https://github.com/user-attachments/assets/e89fb9b4-4c56-478e-8ad8-ea92d4a02562)


### Classification Report

```
               precision    recall  f1-score   support

      benign       0.91      0.97      0.94       382
   malignant       0.95      0.83      0.89       179
      normal       0.95      0.94      0.94       117

    accuracy                           0.93       678
   macro avg       0.94      0.91      0.92       678
weighted avg       0.93      0.93      0.93       678
```

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## Example Usage

The model can be used to classify new images. Below is an example function to classify a single image.

```python
def classify_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = classifier.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = class_labels[predicted_class[0]]

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}")
    plt.show()
```

## Requirements

The project requires the following libraries:

- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

You can install the necessary packages using:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage

1. **Prepare the Dataset**: Organize the dataset into `Train/`, `Valid/`, and `Test/` directories with subfolders for each class.
2. **Train the Model**: Run the provided code to train the model.
3. **Evaluate the Model**: Use the test set to evaluate the model's performance.
4. **Classify Images**: Use the `classify_image` function to classify new images.

## Results

- **Accuracy**: 92.92% on the test set.
- **Confusion Matrix**: Shows the model's performance across all classes.
- **Classification Report**: Detailed precision, recall, and F1-score for each class.

## Conclusion

This project demonstrates the effectiveness of CNNs in classifying breast ultrasound images. The high accuracy and detailed performance metrics indicate that the model is reliable for this type of medical image classification task.

## References

- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Keras Documentation: [https://keras.io/](https://keras.io/)
- scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)

