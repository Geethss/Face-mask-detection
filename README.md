# Face Mask Detection

Developed a real-time face mask detection system using deep learning and computer vision. The system detects whether individuals are wearing masks by analyzing frames from the user's webcam. It uses a pre-trained deep learning model to classify faces as either "WithMask" or "WithoutMask" and provides visual feedback with bounding boxes and labels.

## Features

- **Real-Time Face Detection:** Detect faces instantly from a webcam feed using the MTCNN face detection algorithm.
- **Mask Classification:** Classify detected faces as "WithMask" or "WithoutMask" using a trained deep learning model.
- **Bounding Box & Label:** Draw bounding boxes around faces and display labels indicating mask status.
- **Webcam Integration:** Open the webcam for real-time detection and analysis.
- **Colored Indicators:** Use green for "WithMask" and red for "WithoutMask" to easily visualize mask compliance.

## Technologies Used

- **TensorFlow/Keras:** For building and loading the face mask classification model.
- **OpenCV:** To capture webcam input and render real-time video with annotations.
- **MTCNN:** Multi-task Cascaded Convolutional Networks for detecting faces in the webcam feed.
- **NumPy:** Used for reshaping and preprocessing images.
- **HDF5:** To load the pre-trained model from a saved file for mask detection.

This real-time system is ideal for monitoring mask compliance in settings like offices, stores, and public spaces, helping ensure safety protocols are followed.
