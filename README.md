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

### Using Pre-Trained Weights to Make a Prediction:
To run the model on a single file, you need to clone the repo using the instructions above then, from the root directory, run:
```bash
python -m src.inference_CNN <path to an mp3 file>
```
Results will look like this:
```bash
{
  "name": "/your/path/to/the/file/bradmondonyc_0851551530_musicaldown.com_37k_22050_1ch.mp3",
  "Percent_AI": 0.0,
  "Percent_Human": 100.0,
  "Prediction": "human"
}
```
To run the model on an entire directory, you need to adjust the following code in inference_CNN.py:
```python
     human_dirs = [
          
          ["data/human_split", "split human set"],
     ]

     ai_dirs = [
          
          ["data/ai_split", "split AI set"],
     ]

```
Place the path to your directory in either human_dirs or ai_dirs. This was initially built to test how it performed on known human or ai files so the final_results in the CNN_logs will not be accurate. However, the console will print out the prediction for each file in the directory upon running this. Once you have updated the paths above, you just need to run:

```bash
python -m src.inference_CNN
```

You can do the exact same thing with the vision transformer, pointing to a single file or an entire directory by running:
```bash
python -m src.inference_ViT <path to your file>
```
or
```bash
python -m src.inference_ViT
```
After adjusting the paths in inference_ViT.py.



## Training

### Dataset & Data Preparation:

#### A Note on Dataset and Data Quality For Training:
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

#### A Note on Dataset Size:
Our pretrained weights were trained on a total of 600 hours of audio, 300 for human and 300 for AI. We have found that you can get similar performance with just 7.5 hours per class or 15 hours total for the convolutional network, while the vision transformer benefits from at least 25 hours per class to get the same performance. 

### Preprocessing Your Dataset.
Your dataset requires 4 directories:
* Human Training: A directory of mp3s that are human generated and used for training. 
* AI Training: A directory of mp3s that are AI generated and used for training.
* Human Validation: A directory of mp3s that are human generated used for validation. 
* AI Validation: A directory of mp3s that are AI generated used for validation. 

#### Generate the Base File Structure:
Once you have compiled your dataset, run the following:
```bash
python -m data.generate_dirs
```

This will generate the following file structure:
```bash
/data

|----ai_full/
|    |-- ***put your ai-generated mp3s of any length and quality here.***
|----ai_converted/
|    |-- Your original ai files converted to 16k 48kbps
|----ai_split/
|    |-- Your converted ai files split into 3 second clip. 
|----human_full/
|    |-- ***put your ai-generated mp3s of any length and quality here.***
|----human_converted/
|    |-- Your original human files converted to 16k 48kbps
|----human_split/
|    |-- Your converted human files split into 3 second clip.
|----validation_set/
     |-- ai_full/ ***put your ai validation mp3s here***
     |-- ai_converted/
     |-- ai_split/
     |-- human_full/ ***put your human validation mp3s here***
     |-- human_converted/
     |-- human_split/

```
Add your files where noted above. 

#### Converting and Splitting the Data:
 
We then need to prepare your data for the model. Both the Vit and the CNN require the clips to be down sampled to 16k sample rate and 48 bit rate. This helps with generalization. Additionally, the clips need to be split into a consistent size for the model. The below script performs the conversion and splits the clips into 3 second sub-clips.  

```bash
python -m data.convert_and_divide
```

***Your data is now ready for training.***


### Training & Evaluation
To train a model on your dataset, do the following. 

**For CNN:**
```bash
python -m src.train_CNN
```

**For Vision Transformer:**
```bash
python -m src.train_vision_transformer
```

The vision transformer will take significantly longer than the CNN to train

## License 
MIT License

Copyright (c) [2024] [Ai-SPY-Deepfake-Detection-Model]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
