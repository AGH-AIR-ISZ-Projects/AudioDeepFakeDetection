import librosa
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os

counter_lock = multiprocessing.Lock()

root_path = "audio_real/"   #this is path to your audio, change accordingly
destination_folder = "spectrograms/real_noise/"     # this is where spectrograms will be dumped

batch_size = 15  # number of spectrograms to generate before writing to file
counter = 0  # counter to keep track of number of spectrograms generated in current batch

def generate_spectrogram(filename):#, length):
    global counter

    y, sr = librosa.load(root_path+"/"+filename, sr=16000)

    # comment below lines if you DON'T want to add noise to spectrograms
    noise = np.random.normal(0, 0.1, size=len(y))
    noise_y = y + noise

    S = librosa.feature.melspectrogram(y=noise_y, sr=sr)

    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_dB, x_axis='time',
                    y_axis='mel', sr=sr,
                    fmax=16000)

    plt.axis('off')
    plt.savefig(destination_folder+filename[:-4]+"png" ,bbox_inches='tight',transparent=True, pad_inches=0)

    with counter_lock:
        counter += 1  # increment counter
    if counter == batch_size:  # if we've generated enough spectrograms for a batch
        with counter_lock:
            counter = 0  # reset counter
        plt.close('all')  # close all open figures to free up memory

if __name__ == '__main__':
    
    filenames = [filename for filename in os.listdir(root_path)]
    pool = multiprocessing.Pool()
    pool.starmap(generate_spectrogram, zip(filenames))