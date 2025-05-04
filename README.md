# Mongolian OCR Service

This service extracts text from Image and returns the extracted dext as a JSON response.

## Project Structure
```
/project-root
│── dataset/                        # Dataset for training character recognition model
│   ├── chars_test.csv              # Test dataset
│   ├── chars_train.csv             # Train dataset
│── model/                          # Recognition model and Encoder
│   ├── encoder.npy                 # Encoder
│   ├── recognition_model.h5        # Trained character recognition model
│── notebook/                       # General notebooks that is used to train and segment
│   ├── demonstration.ipynb         # Demonstration of all process
│   ├── train_recognition.ipynb     # Training recognition model
│── src/                            # Source code for table extraction
│   ├── adjust_height_width.py      # Resize image by finding median height and width of letters
│   ├── post_process.py             # Post-process OCR raw result
│   ├── recognition.py              # Recognition of letter and line
│   ├── segmentation_character.py   # Segment characters from word
│   ├── segmentation_contour.py     # Segment contours from input image
│   ├── segmentation_line.py        # Segment lines from contour
│   ├── segmentation_word.py        # Segment words from line
│   ├── service.py                  # Main table extraction logic
│   ├── skew_correction.py          # Correct skew angle of input image
│   ├── utils.py                    # Utility functions
│── test_imgs/                      # Images to test OCR service
│   ├── img_001.jpg                 # Main table extraction logic
│   ├── ...                    
│── main.py                         # FastAPI application
│── requirements.txt                # Dependencies
│── Dockerfile                      # Docker configuration
│── README.md                       # Documentation
│── requist.ipynb                   # Example requist
```

## Setup Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/batzorich/Mongolian-OCR.git
cd Mongolian-OCR
```

### 2. Create and Activate Virtual Environment
```sh
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run from Notebook
Open the demonstration notebook at `notebook/demonstration.ipynb`.  
Set the image path in the notebook (e.g., `../test_imgs/img_001.jpg`).  
Run the cells to extract and display the recognized text.


## Docker Setup

> ⚠️ If Docker is not installed, follow the [official Docker installation guide](https://docs.docker.com/get-docker/) for your OS.

### 1. Build the Docker Image
```sh
docker build --no-cache -t text-extraction .
```

### 2. Run the Container
```sh
docker run -p 8000:8000 text-extraction
```

## API Endpoints

### Health Check
```http
GET /health
```
Response:
```json
{"status": "ok"}
```

### Extract Text
```http
POST /extract-text
```
#### Request
- **Content-Type:** multipart/form-image
- **File:** JSON file containing image

#### Response
```json
{
    "message": "Text extracted successfully",
    "data": "4.2.Төрийн болон орон нутгийн өмчийн орон сууцны санг бүрдүүлж,..."
}
```