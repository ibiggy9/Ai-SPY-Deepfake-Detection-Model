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



def load_model(model_path, patch_size=32, embedding_dim=256, num_heads=8, num_layers=4, num_classes=2):
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

def preprocess_audio(audio_path, sr=16000, duration=3):
    y, _ = librosa.load(audio_path, sr=sr)
    clips = [y[i:i + sr * duration] for i in range(0, len(y) - sr * duration + 1, sr * duration)]
    processed_clips = []

    for clip in clips:
        D = librosa.stft(clip, n_fft=1024, hop_length=256, win_length=1024, window='hann')
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        epsilon = 1e-6
        S_db = (S_db - np.mean(S_db)) / (np.std(S_db) + epsilon)
        S_db = S_db[:, :100 * duration]

        patch_size = 32
        stride = 8
        patches = []
        for i in range(0, S_db.shape[1] - patch_size + 1, stride):
            for j in range(0, S_db.shape[0] - patch_size + 1, stride):
                patch = S_db[j:j + patch_size, i:i + patch_size]
                patches.append(patch.flatten())
        patches = np.array(patches)
        spectrogram_tensor = torch.tensor(patches, dtype=torch.float32)
        processed_clips.append(spectrogram_tensor)

    return processed_clips

def predict_neural_for_testing(clips, model):
    model.eval()
    results = {'chunk_results': []}
    overall_probs = []

    with torch.no_grad():
        for i, clip in enumerate(clips):
            try:
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
                print(e)

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

def process_audio_file(audio_file, model, bit_rate, sample_rate):
    try:
        audio_file_path = f"data/Vit_Logs/{uuid.uuid4()}.mp3"
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
    args = [(audio_file, model, bit_rate, sample_rate) for audio_file in audio_files]

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

def run_models(model, dir, isHuman, shortName, bit_rate, sample_rate):
    model_path = "data/models"
    try:
        model_path = os.path.join(model_path, model)
        print(model_path)
        output_file_path = f'data/Vit_Logs/results_{shortName}_model.txt'
        final_results_path = f'data/Vit_Logs/final_results_model.txt'
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
        print(e)

def main():
    sample_rate_model = 16000
    bit_rate_model = '48k'

    try:
        os.remove("data/Vit_Logs/final_results_model.txt")
    except Exception as e:
        pass
    
    human_dirs = [
        ["data/human_full", "full human set"],
        ["data/human_converted", "converted human set"],
        ["data/human_split", "split human set"],
    ]

    ai_dirs = [
        ["data/ai_full", "full AI set"],
        ["data/ai_converted", "converted AI set"],
        ["data/ai_split", "split AI set"],
    ]

    def signal_handler(sig, frame):
        print('KeyboardInterrupt is caught')
        clean_up()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        for model in os.listdir("data/models"):
            print(f"Running model: {model}")
            with open("data/Vit_Logs/final_results_model.txt", "a") as file:
                file.write(f"\n\nMODEL: {model}\n")
                file.write(f"HUMAN RESULTS\n")
            start_time = time.time()
            for directory in human_dirs:
                run_models(model=model, dir=directory[0], isHuman=True, shortName=directory[1], sample_rate=sample_rate_model, bit_rate=bit_rate_model)
            print(f"Time taken for human: {time.time() - start_time}")
            ai_start_time = time.time()
            with open("data/Vit_Logs/final_results_model.txt", "a") as file:
                file.write("\nAI RESULTS\n")
            for directory in ai_dirs:
                run_models(model=model, dir=directory[0], isHuman=False, shortName=directory[1], sample_rate=sample_rate_model, bit_rate=bit_rate_model)
            print(f"Time taken for ai: {time.time() - ai_start_time}")
            print(f"Time taken for both: {time.time() - start_time}")
        clean_up()
    except KeyboardInterrupt:
        clean_up()

if __name__ == '__main__':
    main()