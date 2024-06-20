from joblib import Parallel, delayed
import librosa
import numpy as np
import glob
import time
import os

def process_file(file, sr=16000, duration=3):
    try:
        y, _ = librosa.load(file, sr=sr, duration=duration)
        y = librosa.util.fix_length(y, size=sr * duration)
        S = np.abs(librosa.stft(y))**2
        S_db = librosa.power_to_db(S, ref=np.max)
        return np.mean(S_db), np.std(S_db), S_db.size
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def calculate_global_mean_std(files, sr=16000, duration=3, n_jobs=24):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(file, sr, duration) for file in files
    )

    mean_sum = 0
    std_sum = 0
    count = 0

    for result in results:
        if result is not None:
            mean, std, size = result
            mean_sum += mean * size
            std_sum += std * size
            count += size

    global_mean = mean_sum / count
    global_std = std_sum / count

    return global_mean, global_std

def save_mean_std_to_file(global_mean, global_std, output_file):
    with open(output_file, 'w') as f:
        f.write(f"global_mean: {global_mean}\n")
        f.write(f"global_std: {global_std}\n")

if __name__ == "__main__":
    ai_directory = '/data/ai_split'
    human_directory = '/data/human_split'
    output_file = '/data/global_stats.txt'

    ai_files = glob.glob(os.path.join(ai_directory, '*.mp3'))
    human_files = glob.glob(os.path.join(human_directory, '*.mp3'))
    all_files = ai_files + human_files

    start_time = time.time()

    global_mean, global_std = calculate_global_mean_std(all_files)

    end_time = time.time()

    print(f"Processing time: {end_time - start_time} seconds")
    print(f"Global mean: {global_mean}, Global std: {global_std}")

    save_mean_std_to_file(global_mean, global_std, output_file)
