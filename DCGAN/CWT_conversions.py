from audio2numpy import open_audio
import os
import glob
import pickle
import pywt
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from wavelets import WaveletAnalysis
import scipy
import io
import scipy.io.wavfile
import pydub

"""
Script for testing the Continuous Wavelet Transform (CWT), inverse CWT, making plots
"""

path = "./../birdsong-recognition/train_audio"

def mp3ToFreqDom(path, filter_value):
    # input: path to mp3 file, filter_value to remove silence in mp3
    # output: numpy array of size 28x28

    signal, sampling_rate = open_audio(path)
    print(signal.shape)  # returns a long vector
    print("Sampling rate = ", sampling_rate)
    # filter signal
    signal_filtered = signal[signal > filter_value]
    print(signal_filtered.shape)
    # scales determines the dimension of the frequency (y-axis):
    coef, freq = pywt.cwt(signal_filtered, scales=np.arange(1, 29), wavelet='gaus1')
    print(coef.shape)
    # print(coef[:,1:28])
    return coef[:,1:28]


def mp3_to_signal(path, sampling_rate):
    ### mp3 to numpy array using AudioSegment, specify sampling rate:
    # f_s = 44100  = original sampling (frame) rate = # measurements per second
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(sampling_rate)
    sample = audio.get_array_of_samples()
    audio_arr = np.array(sample).T.astype(np.float32)
    audio_arr /= np.iinfo(sample.typecode).max
    return audio_arr

def signal_to_cwt(original_signal, sampling_rate):
    ### Compte CWT using wavelets package:
    # pip install git+https://github.com/aaren/wavelets
    #x = audio_arr[:400] #cut the signal to our input width

    # other continuous wavelet transformer:
    #wa= WaveletAnalysis(x,dt=dt) # wavelet object
    #power=wa.wavelet_power #wavelet power spectrum = the coefficients in cwt function
    #plt.matshow(power)
    #plt.show()
    #scales=wa.scales # 62 scales (for some reason...)
    #t=wa.time # associated time vector: 0 to 4, of length 400
    #rx=wa.reconstruction() # reconstruction of original data, needs wavelet object as input!

    ### Compute CWT using pywt package:
    scales = np.arange(1, 33)
    coef, freq = pywt.cwt(original_signal, scales=scales, wavelet='gaus1', sampling_period=1/sampling_rate)
    return coef

def cwt_to_signal(coef, sampling_rate):
    dt = 1/sampling_rate  # sample spacing
    scales = np.arange(1, 33)
    ### Reconstruction using https://github.com/PyWavelets/pywt/issues/328 (doesnt work well):
    # Inverse CWT:
    mwf = pywt.ContinuousWavelet('gaus1').wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]
    r_sum = np.transpose(np.sum(np.transpose(coef) / scales ** 0.5, axis=-1))
    reconstructed = r_sum * (dt ** 0.5 / y_0)
    # print(reconstructed.shape)
    return reconstructed

def signal_to_mp3(original_signal, signal, sampling_rate, filename):
    # parse to audio:
    wav_io = io.BytesIO()
    # original audio array:
    scipy.io.wavfile.write(wav_io, sampling_rate, original_signal)
    wav_io.seek(0)
    sound = pydub.AudioSegment.from_wav(wav_io)
    sound.export(f"./../train_birds/{filename}_{sampling_rate}_original.mp3", format="mp3")

    signal = np.array(signal).astype(np.float32)
    scipy.io.wavfile.write(wav_io, sampling_rate, signal)
    wav_io.seek(0)
    sound = pydub.AudioSegment.from_wav(wav_io)
    sound.export(f"./../train_birds/{filename}_{sampling_rate}_reconstructed.mp3", format="mp3")

def get_gen_im():
    file = open("./../generated_imgs.pkl", "rb")
    coeffs = pickle.load(file)
    im1 = coeffs[1] # get first image
    #im1 = im1[:,:,0]
    im1 = im1[1:33, :, 1] #downsample
    return im1
generated_image = get_gen_im()
generated_signal = cwt_to_signal(generated_image, 400)
# parse to mp3:
# parse to audio:
wav_io = io.BytesIO()
generated_signal = np.array(generated_signal).astype(np.float32)
scipy.io.wavfile.write(wav_io, 400, generated_signal)
wav_io.seek(0)
sound = pydub.AudioSegment.from_wav(wav_io)
sound.export(f"./../generated_sound.mp3", format="mp3")


#fp = "./../birdsong-recognition/train_audio/aldfly/XC2628.mp3"
fp = "./../birdsong-recognition/train_audio/ameavo/XC99571.mp3"
#fp = "./../birdsong-recognition/train_audio/aldfly/XC16967.mp3"
filename = "XC99571"

#original44100 = mp3_to_signal(fp, 44100)
#coef44100 = signal_to_cwt(original44100, 44100)
#reconstructed44100 = cwt_to_signal(coef44100, 44100)
#signal_to_mp3(original44100, reconstructed44100, 44100, filename)

#original4000 = mp3_to_signal(fp, 4000)
#coef4000 = signal_to_cwt(original4000, 4000)
#reconstructed4000 = cwt_to_signal(coef4000, 4000)
#signal_to_mp3(original4000, reconstructed4000, 4000, filename)

#riginal400 = mp3_to_signal(fp, 400)
#coef400 = signal_to_cwt(original400, 400)
#reconstructed400 = cwt_to_signal(coef400, 400)
#signal_to_mp3(original400, reconstructed400, 400, filename)

def plot_coefs():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    ax1.matshow(coef44100[:, :1200])
    ax1.set_title("CWT coefficients at sampling rate 44100")
    ax2.matshow(coef4000[:, :600])
    ax2.set_title("CWT coefficients at sampling rate 4000")
    ax3.matshow(coef400[:, :600])
    ax3.set_title("CWT coefficients at sampling rate 400")
    plt.show()

#plot_coefs()

def plot_signal_reconstructed(original, reconstructed, sampling_rate):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    # Plot original signal
    T = len(original)
    ax1.plot(np.arange(1,T+1), original)
    ax1.set_title(f"Original signal, at sampling rate {sampling_rate}")
    ax2.plot(np.arange(1,T+1), reconstructed)
    ax2.set_title(f"Reconstructed signal, at sampling rate {sampling_rate}")
    plt.show()
#plot_signal_reconstructed(original44100, reconstructed44100, 44100)
#plot_signal_reconstructed(original4000, reconstructed4000, 4000)
#plot_signal_reconstructed(original400, reconstructed400, 400)

def plot_coef_at_diff_fs():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    ax1.matshow(coef44100[:,:600])
    ax1.set_title(f"CWT coefficients at sampling rate = 44100")
    ax2.matshow(coef4000[:, :600])
    ax2.set_title(f"CWT coefficients at sampling rate = 4000")
    ax3.matshow(coef400[:, :600])
    ax3.set_title(f"CWT coefficients at sampling rate = 400")
    plt.show()
#plot_coef_at_diff_fs()


def plot_signal_reconstr_at_diff_fs():
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True, figsize=(12, 8))
    # Plot original signal
    T = len(original44100)
    ax1.plot(np.arange(1,T+1), original44100)
    ax1.set_title(f"Original signal, at sampling rate = 44100")
    ax2.plot(np.arange(1,T+1), reconstructed44100)
    ax2.set_title("Reconstructed signal, at sampling rate = 44100")

    # Plot original signal
    T = len(original4000)
    ax3.plot(np.arange(1,T+1), original4000)
    ax3.set_title(f"Original signal, at sampling rate = 4000")
    ax4.plot(np.arange(1,T+1), reconstructed4000)
    ax4.set_title("Reconstructed signal, at sampling rate = 4000")

    # Plot original signal
    T = len(original400)
    ax5.plot(np.arange(1,T+1), original400)
    ax5.set_title(f"Original signal, at sampling rate = 400")
    ax6.plot(np.arange(1,T+1), reconstructed400)
    ax6.set_title("Reconstructed signal, at sampling rate = 400")
    plt.show()
#plot_signal_reconstr_at_diff_fs()

def plot_signal_coef_reconstructed(original, coef, reconstructed, sampling_rate):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    T = len(original)
    # Plot original signal
    ax1.plot(original)
    # ax1.plot(np.arange(1,401), x)
    ax1.set_title(f"Original signal, at sampling rate = {sampling_rate}")
    # Plot CWT:
    # ax2.matshow(power)
    ax2.matshow(coef[:, :200])
    ax2.set_title("Scalogram of CWT")

    # ax3.plot(np.arange(1,401), rx)
    ax3.plot(reconstructed)
    ax3.set_title("Reconstructed signal")
    plt.show()
#plot_signal_coef_reconstructed(original44100, coef44100, reconstructed44100, 44100)