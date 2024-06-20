# Audio Deepfake Detection

## Overview 
This project foucses on detecting audio deep fakes with deep learning

## Installation
To get started, make sure you have Python 3.10.6 installed. You can check your Python version with:
```bash
python --version
```
If you don't have Python 3.10.6 installed, you can download it from [python.org](https://www.python.org). 

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/audio-deepfake-detection.git
cd audio-deepfake-detection
pip install -r requirements.txt
```

## Usage

### Using Pre-Trained weights.

### A few notes on dataset and data quality. 
This repo does not come with sample data. To train your own model, you need to supply your own dataset. These can be found all over the internet, but this one is fairly robust [this data set](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition). 

It is important to note however, that for practical applications, the quality of your dataset is very important. Both convolutional neural networks and vision transformers can generalize to a degree, but each deepfake model has a slightly different signal and they are reducing in strength as these models become more sophisticated. Consistently revisiting your dataset, how recently the data has been produced and the total composition of the dataset. 

Ideally, you want the AI half of your dataset to be comprised of the latest and most performant deepfake models in equal balance. You should also seek as many "in the wild" examples from the internet as possible. These deepfakes often contain important additional noise such as background music and stem mixtures that come from exporting audio with professional editing software. Our experiementation suggests that you ideally want at least 50% of your dataset to be high quality and convincing "in the wild" examples and 50% to be raw outputs of the latest most performance audio deepfake models. 

On the human side of the dataset, the helpful ontological concept is production type, i.e. professional recordings, single speaker auration, highly edited productions, outdoor recordings, multiple languages, accents, and conditions. We found that over emphasizing any one of these led to an unhelpful bias in the model. Some examples of samples we used in our most successful attempts were: 
* News clips
* Sporting clips
* Low quality indoor hot mics
* High quality indoor recordings
* Lectures
* Podcasts
* Online gaming streams
* Multiple speaker news panels that feature frequent interruption
* Outdoor recordings of any kind (applause and wind noise really help bolster generalization)
* Recordings with echo or reverb

#### A note on dataset size
Our pretrained weights were trained on a total of 600 hours of audio, 300 for human and 300 for AI. We have found that you can get similar performance with just 7.5 hours per class or 15 hours total for the convolutional network, while the vision transformer benefits from at least 25 hours per class to get the same performance. 

### Data Preparation
Once you have compiled your dataset, put those in a subdirectory in the data directory called "ai_full" and "human_full" respectively. 
```bash
/data
|----ai_full/
|    |-- ai-generated mp3s of any length and quality.
|----human_full/
|    |-- human-generated mp3s of any length and quality.
```

Then run:
```bash
python -m data.convert_and_divide.py
```

### Training & Evaluation
ab

### Inference

## Contributing 

## License 

## Acknowledgements