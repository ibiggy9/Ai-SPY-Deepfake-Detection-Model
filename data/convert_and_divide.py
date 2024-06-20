import os
from multiprocessing import Pool, cpu_count
from pydub import AudioSegment
from tqdm import tqdm
from joblib import Parallel, delayed
import librosa
import numpy as np
import math
import glob
import shutil
import time

class AudioConverter:
    """
    A class to convert audio files to a specific bit rate and sample rate. Note that this must be done to prepare your data for the models. 

    Attributes:
    ----------
    input_dir : str
        The directory containing the input audio files.
    output_dir : str
        The directory where the converted audio files will be saved.
    new_sample_rate : int
        The new sample rate for the audio files.
    new_bit_rate : int
        The new bit rate for the audio files.

    Methods:
    -------
    convert_audio(args):
        Converts a single audio file to the specified bit rate and sample rate.
    find_audio_files(directory):
        Finds all audio files in the specified directory.
    process_files():
        Processes all audio files in the input directory, converting them to the specified bit rate and sample rate.
    """
    
    def __init__(self, input_dir, output_dir, new_sample_rate, new_bit_rate):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.new_sample_rate = new_sample_rate
        self.new_bit_rate = new_bit_rate

    def convert_audio(self, args):
        input_file, output_dir, new_sample_rate, new_bit_rate = args
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_frame_rate(new_sample_rate)
        audio = audio.set_sample_width(2)  # 2 bytes for 16-bit audio
        audio.export(output_file, format="mp3", bitrate=f"{new_bit_rate}k")

    def find_audio_files(self, directory):
        extensions = ['.mp3', '.wav', '.flac', '.mp4']
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    yield os.path.join(root, file)

    def process_files(self):
        audio_files = list(self.find_audio_files(self.input_dir))
        convert_args = [(file, self.output_dir, self.new_sample_rate, self.new_bit_rate) for file in audio_files]
        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(self.convert_audio, convert_args), total=len(convert_args)):
                pass

class AudioSplitter:
    """
    A class to split audio files into smaller chunks.

    Attributes:
    ----------
    input_dir : str
        The directory containing the input audio files.
    output_dir : str
        The directory where the split audio files will be saved.
    split_length : int
        The length of each split chunk in seconds.
    bit_rate : str
        The desired bit rate for the output audio files.

    Methods:
    -------
    split_song(args):
        Splits a single audio file into smaller chunks.
    process_directory():
        Processes all audio files in the input directory, splitting them into chunks.
    """
    
    def __init__(self, input_dir, output_dir, split_length, bit_rate):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.split_length = split_length
        self.bit_rate = bit_rate

    def split_song(self, args):
        try:
            filename, split_length, output_dir, bit_rate = args
            song = AudioSegment.from_file(filename)
            song_length_in_sec = len(song) // 1000
            num_chunks = math.ceil(song_length_in_sec / split_length)
            base_name = os.path.splitext(os.path.basename(filename))[0]

            for i in range(num_chunks):
                try:
                    start_time = i * split_length * 1000
                    end_time = (i + 1) * split_length * 1000
                    chunk = song[start_time:end_time]
                    chunk.export(os.path.join(output_dir, f"{base_name}_chunk_{i}.mp3"), format="mp3", bitrate=bit_rate)
                except Exception as e:
                    print(f"Failed to export chunk {i}: {e}")
        except Exception as e:
            print(f"Failed to process the file {filename}: {e}")

    def process_directory(self):
        if not os.path.exists(self.output_dir):
            print("Making Directory:", self.output_dir)
            os.mkdir(self.output_dir)

        tasks = []
        for filename in os.listdir(self.input_dir):
            tasks.append((os.path.join(self.input_dir, filename), self.split_length, self.output_dir, self.bit_rate))
            
        with Pool(cpu_count()) as p:
            list(tqdm(p.imap(self.split_song, tasks), total=len(tasks)))


def run_audio_conversion_and_splitting(input_dir, converted_dir, split_dir):
    """
    Runs the audio conversion and splitting processes on the specified directories.
    """
 

    # Convert audio files
    converter = AudioConverter(input_dir, converted_dir, new_sample_rate=16000, new_bit_rate=48)
    converter.process_files()

    # Split converted audio files
    splitter = AudioSplitter(converted_dir, split_dir, split_length=3, bit_rate="48k")
    splitter.process_directory()

if __name__ == "__main__":
    input_dir_ai = './data/ai_full'
    converted_dir_ai = './data/ai_converted'
    split_dir_ai = './data/ai_split'

    input_dir_human = './data/human_full'
    converted_dir_human = './data/human_converted'
    split_dir_human = './data/human_split'

    run_audio_conversion_and_splitting(input_dir=input_dir_ai, converted_dir=converted_dir_ai, split_dir=split_dir_ai)  
    run_audio_conversion_and_splitting(input_dir=input_dir_human, converted_dir=converted_dir_human, split_dir=split_dir_human)
