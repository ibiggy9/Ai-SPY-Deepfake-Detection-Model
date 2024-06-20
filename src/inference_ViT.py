import os
import torch
import json
import time
from multiprocessing import Pool
from tqdm import tqdm
from pydub import AudioSegment
import uuid
import torch.nn.functional as F
import librosa
import glob
import numpy as np
import signal
from models.vit_model import VisionTransformer
import argparse

def load_model(model_path, patch_size=16, embedding_dim=512, num_heads=8, num_layers=8, num_classes=2):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VisionTransformer(patch_size, embedding_dim, num_heads, num_layers, num_classes, device=device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_audio(audio_path, sr=16000, duration=3, global_mean=-58.18715250929163, global_std=15.877255962380845):
    y, _ = librosa.load(audio_path, sr=sr)
    y = librosa.util.fix_length(y, size=sr * duration)
    y = np.clip(y, -1.0, 1.0)
    clips = [y[i:i + sr * duration] for i in range(0, len(y) - sr * duration + 1, sr * duration)]
    
    processed_clips = []
    for clip in clips:
        S = np.abs(librosa.stft(clip))**2
        S_db = librosa.power_to_db(S + 1e-10, ref=np.max)
        S_db = (S_db - global_mean) / global_std

        target_shape = (1025, 94)
        if S_db.shape != target_shape:
            S_db = np.pad(S_db, (
                (0, max(0, target_shape[0] - S_db.shape[0])), 
                (0, max(0, target_shape[1] - S_db.shape[1]))
            ), mode='constant', constant_values=global_mean)
            S_db = S_db[:target_shape[0], :target_shape[1]]
        spectrogram_tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0)
        processed_clips.append(spectrogram_tensor)
        
    return processed_clips

def predict_neural_for_testing(clips, model):
    model.eval()
    results = {'chunk_results': []}
    overall_probs = []

    with torch.no_grad():
        for i, clip in enumerate(clips):
            try:
                print(f"Processing clip {i+1}/{len(clips)}: shape={clip.shape}, dtype={clip.dtype}")
                output = model(clip.unsqueeze(0))
                probs = F.softmax(output, dim=1)
                probability_ai = round(probs[0][0].item() * 100, 2)
                prediction = output.argmax(dim=1).item()

                if probability_ai >= 50:
                    predicted_label = 'ai'
                    confidence = probability_ai
                else:
                    predicted_label = 'human'
                    confidence = 100 - probability_ai

                chunk_result = {
                    "chunk": i + 1,
                    "prediction": predicted_label,
                    "confidence": f"{confidence:.2f}%",
                    "Probability_ai": f"{probability_ai:.2f}%"
                }
                results['chunk_results'].append(chunk_result)
                overall_probs.append(probability_ai)
            except Exception as e:
                print(f"Error processing clip {i+1}: {e}")

        ai_chunk_count = sum(1 for result in results['chunk_results'] if result['prediction'] == 'ai')
        percentage_ai_chunks = (ai_chunk_count / len(clips)) * 100
        print(f"Percent AI chunks {percentage_ai_chunks}")

        if percentage_ai_chunks >= 50:
            overall_prediction = 'ai'
        else:
            overall_prediction = 'human'

        print(overall_prediction)
        results.update({
            "status": "success",
            "prediction": overall_prediction,
        })

        return results
    
def convert_to_mp3(input_file, output_file, bit_rate="", sample_rate=16000):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="mp3", bitrate=bit_rate, parameters=["-ar", str(sample_rate)])
    return output_file

def process_audio_file(audio_file, model, sample_rate, bit_rate):
    try:
        audio_file_path = f"Vit_Logs/{uuid.uuid4()}.mp3"
        converted_audio = convert_to_mp3(audio_file, audio_file_path, bit_rate=bit_rate, sample_rate=sample_rate)
        clips = preprocess_audio(converted_audio, sr=sample_rate)

        result = predict_neural_for_testing(clips, model)
        os.remove(audio_file_path)

        ai_count = sum(1 for result in result['chunk_results'] if result['prediction'] == "ai")
        human_count = sum(1 for result in result['chunk_results'] if result['prediction'] == "human")
        total = len(result['chunk_results'])

        return {
            "name": audio_file,
            "Percent_AI": (ai_count / total) * 100,
            "Percent_Human": (human_count / total) * 100,
            "Prediction": "ai" if ai_count > human_count else "human",
        }
    except Exception as e:
        print(f"Error processing audio file: {e}")

        return {
            "name": audio_file,
            "Percent_AI": 1,
            "Percent_Human": 0,
            "Prediction": "ai",
        }

def process_directory(directory_path, model, sample_rate, bit_rate):
    mp3_files = glob.glob(os.path.join(directory_path, '*'))
    audio_files = mp3_files
    print(f"Current Dir {directory_path}")
    args = [(audio_file, model, sample_rate, bit_rate) for audio_file in audio_files]

    with Pool(processes=24) as pool:
        result_objects = pool.starmap_async(process_audio_file, args)
        results = []
        for result in tqdm(result_objects.get(), total=len(args)):
            results.append(result)
    print(f"Result: {results}")
    return results

def save_results_to_file(results, output_file_path):
    with open(output_file_path, 'w') as file:
        for result in results:
            try:
                json.dump(result, file)
                file.write('\n')
            except Exception as e:
                print(f"Error writing result to file: {e}")

def run_models(model, dir, isHuman, shortName, sample_rate, bit_rate):
    model_path = "./models/pretrained_weights/Vit_Ai-SPY.pth"
    try:
        print(model_path)
        output_file_path = f'Vit_Logs/results_{shortName}_model.txt'
        final_results_path = f'Vit_Logs/final_results_model.txt'
        print("Loading Model")
        model = load_model(model_path)
        results = process_directory(dir, model, sample_rate=sample_rate, bit_rate=bit_rate)
        save_results_to_file(results, output_file_path)

        ai_error = []
        human_error = []
        predictions = []
        correct_predictions = 0
        
        with open(output_file_path, "r") as file:
            for line in file:
                data = json.loads(line.strip())
                percent_ai = float(data["Percent_AI"])
                percent_human = float(data["Percent_Human"])
                
                # Calculate errors as percentages of total clips
                if data["Prediction"] == "ai" and percent_ai < 100:
                    ai_error.append(100 - percent_ai)
                elif data["Prediction"] == "human" and percent_human < 100:
                    human_error.append(100 - percent_human)
                
                predictions.append(data["Prediction"])
                if (isHuman and data["Prediction"] == "human") or (not isHuman and data["Prediction"] == "ai"):
                    correct_predictions += 1

        total_predictions = len(predictions)
        percent_correct = (correct_predictions / total_predictions) * 100

        # Calculate average errors
        average_ai_error = sum(ai_error) / total_predictions if total_predictions else 0
        average_human_error = sum(human_error) / total_predictions if total_predictions else 0

        # Calculate total average error
        total_average_error = (sum(ai_error) + sum(human_error)) / total_predictions if total_predictions else 0

        print(f"\nResults for {shortName}:")
        with open(final_results_path, "a") as file:
            file.write(f"\n\nRESULTS FOR {shortName.upper()} Model\n")
            file.write(f"Average AI Error: {average_ai_error:.2f}\n")
            file.write(f"Average Human Error: {average_human_error:.2f}\n")
            file.write(f"Total Average Error: {total_average_error:.2f}\n")
            file.write(f"Total Predictions Correct: {correct_predictions} / {total_predictions}\n")
            file.write(f"Percent Correct: {percent_correct:.2f}%\n")

        if ai_error or human_error:
            print(f"Total AI Error: {sum(ai_error)}")
            print(f"Average AI Error: {average_ai_error:.2f}")
            print(f"Total Human Error: {sum(human_error)}")
            print(f"Average Human Error: {average_human_error:.2f}")
            print(f"Total Error: {sum(human_error) + sum(ai_error)}")
            print(f"Total Average Error: {total_average_error:.2f}")
            print(f"Percent Correct: {percent_correct}%")
        else:
            print("No data to calculate error.")

    except Exception as e:
        print("error 2")
        print(e)

def main():
    parser = argparse.ArgumentParser(description='Run inference on audio files using a Vision Transformer model.')
    parser.add_argument('path', type=str, nargs='?', default=None, help='Path to the audio file or directory of audio files.')
    parser.add_argument('--model', type=str, default='./models/pretrained_weights/Vit_Ai-SPY.pth', help='Path to the model file.')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio processing.')
    parser.add_argument('--bit_rate', type=str, default='48k', help='Bit rate for audio conversion.')

    args = parser.parse_args()
    model_path = args.model
    sample_rate = args.sample_rate
    bit_rate = args.bit_rate
    path = args.path

    human_dirs = [
        
        ["data/human_split", "split human set"],
    ]

    ai_dirs = [
        
        ["data/ai_split", "split AI set"],
    ]

    def signal_handler(sig, frame):
        print('KeyboardInterrupt is caught')
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        os.remove("./Vit_Logs/final_results_model.txt")
    except Exception as e:
        pass

    print(f"Loading Model: {model_path}")
    model = load_model(model_path)

    if path:
        if os.path.isfile(path):
            result = process_audio_file(path, model, sample_rate, bit_rate)
            print(json.dumps(result, indent=2))
        elif os.path.isdir(path):
            results = process_directory(path, model, sample_rate, bit_rate)
            output_file_path = f'Vit_Logs/results_{uuid.uuid4().hex}.txt'
            save_results_to_file(results, output_file_path)
            print(f"Results saved to {output_file_path}")
        else:
            print(f"The path {path} is neither a file nor a directory. Please provide a valid path.")
    else:
        for directory in human_dirs + ai_dirs:
            isHuman = "human" in directory[1]
            run_models(model=model_path, dir=directory[0], isHuman=isHuman, shortName=directory[1], sample_rate=sample_rate, bit_rate=bit_rate)

if __name__ == '__main__':
    main()
