# Eye Disease Detection

This project uses deep learning to classify retinal images and detect eye diseases such as cataracts and other conditions.


## Cloning
```bash
git clone --recurse-submodules https://github.com/Mallikapam/Eye_Disease_Classification.git
```

## Installation
```bash
python -m venv retina-env
source retina-env/bin/activate 
pip install -r requirements.txt
```


## Usage
```bash
python eye_disease_analysis.py
```

## Dataset
Images are stored in the following directory structure:

retina_dataset/
├── dataset/
│ ├── 0_cataract/
│ ├── 1_normal/
│ ├── 2_glaucoma/
│ └── 3_retina_disease/

## Model Architecture

The CNN has 3 convolutional layers with batch normalization and max pooling, followed by a fully connected layer. Images are resized and normalized before training.

## Results

We achieved an accuracy of 63.33% on the test dataset. 