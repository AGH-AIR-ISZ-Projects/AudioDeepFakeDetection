import librosa
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os

counter_lock = multiprocessing.Lock()

root_path = "audio_real"
destination_folder = "spectrograms_janek/reals/"

batch_size = 15  # number of spectrograms to generate before writing to file
counter = 0  # counter to keep track of number of spectrograms generated in current batch
NORM_CONSTANT = 2.0 ** (16 - 1)

def generate_spectrogram(filename):#, length):
    global counter
    # if(float(length) >=1 and float(length) <=6):
        # make spectrogram
    y, sr = librosa.load(root_path+"/"+filename, duration=3, offset=0.5,sr=16000)

    signal = np.zeros((int(3*sr)))
    signal[:len(y)] = y

    signal_len = len(signal)
    
    # Generate White Gaussian noise
    noise = np.random.normal(size=(2, signal_len))
    
    # Normalize signal and noise
    signal_norm = signal / NORM_CONSTANT
    noise_norm = noise / NORM_CONSTANT
    
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(15, 30)
    
    # Compute K (covariance matrix) for each noise 
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, 2)) * K  
    
    # Generate noisy signal
    noise_y = signal + K.T * noise

    S = librosa.feature.melspectrogram(y=noise_y,
        sr=sr,
        n_fft=1024,
        win_length=512,
        window="hamming",
        hop_length = 256,
        n_mels=128,
        fmax=sr / 2)

    S_dB = librosa.power_to_db(S, ref=np.max)

    img1 = librosa.display.specshow(S_dB[0, :], x_axis='time',
                    y_axis='mel', sr=sr,
                    fmax=sr / 2)
    
    plt.axis('off')
    plt.savefig(destination_folder+filename[:-5]+"_1."+"png" ,bbox_inches='tight',transparent=True, pad_inches=0)
    
    img2 = librosa.display.specshow(S_dB[1, :], x_axis='time',
                y_axis='mel', sr=sr,
                fmax=sr / 2)

    plt.axis('off')
    plt.savefig(destination_folder+filename[:-5]+"_2."+"png" ,bbox_inches='tight',transparent=True, pad_inches=0)


    with counter_lock:
        counter += 1  # increment counter
    if counter == batch_size:  # if we've generated enough spectrograms for a batch
        with counter_lock:
            counter = 0  # reset counter
        plt.close('all')  # close all open figures to free up memory

if __name__ == '__main__':
    
    filenames = [filename for filename in os.listdir(root_path)]
    filenames = filenames[:5_000] # take only first 5_000
    pool = multiprocessing.Pool()
    pool.starmap(generate_spectrogram, zip(filenames))