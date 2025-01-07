# Mongolian OCR

This project contains a series of scripts and notebooks to perform OCR on Mongolian characters. It includes image preprocessing, segmentations, and prediction processes using a trained model.

## Setup Instructions

### 1. Create a Virtual Environment
To set up the project environment, create a virtual environment using `venv`:

### 3. Install Required Dependencies
Once the virtual environment is activated, install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

This will install all required libraries for the project, including dependencies for OCR and image processing.

## Project Structure

- **notebook/segmentation_and_prediction.ipynb**: A Jupyter notebook containing the complete process of image segmentation and prediction on a sample image.
- **script/**: Folder containing all the necessary Python scripts used in the project for segmentation, prediction, and preprocessing.

## Usage

1. **Segmentation and Prediction**: You can run the segmentation and prediction process on a sample image by executing the cells in `notebook/segmentation_and_prediction.ipynb`.

2. **Scripts**: The Python scripts in the `script/` folder implement individual components of the OCR pipeline:
   - **Segmentation**: Scripts that handle image preprocessing and character segmentation.
   - **Prediction**: The model and prediction logic for recognizing the characters.
