from audio2numpy import open_audio
import os
import glob
import pickle
import pywt
import matplotlib.pyplot as plt
import numpy as np

path = "./../birdsong-recognition/train_audio"

# Todo: fix this function to get all mp3 files for desired number of folders:
def getmp3Files(path, max_folders = 3):
    listOfFolders = os.listdir(path)
    folderList = list()
    counter = 0 # constrain how many bird folders we extract files from
    for folder in listOfFolders:
        completePath = os.path.join(path, folder)
        folderList.append(completePath)
        counter += 1
        if counter == max_folders:
            break
    mp3FilesList = list()

    for f in folderList:

        completePath = os.path.join(f, os.listdir(f))
        mp3FilesList.append(completePath)

    return mp3FilesList

#files = getmp3Files(path)
#print(files)
#print(len(files))


# TODO: change below to work for all bird names in subfolders:
def getmp3FromSubdir(path, bird_name):

    aldfly_dict = {}
    for mp3 in glob.glob(aldly_path):
        print(mp3) # lists full path of all mp3 files in aldfly directory
        signal, sampling_rate = open_audio(mp3) # parse mp3 to numpy array
        aldfly_dict[mp3] = signal
    print(len(aldfly_dict))
    # save dictionary to pickle:
    aldfly_file = open("aldfly.pkl", "wb")
    pickle.dump(aldfly_dict, aldfly_file)
    aldfly_file.close()

    aldfly_file=open("aldfly.pkl", "rb")
    output = pickle.load(aldfly_file)
    print(len(output))
    print(output)

#aldly_path = "./../birdsong-recognition/train_audio/aldfly/*.mp3"
#getmp3FromSubdir(aldly_path)

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


fp = "./../birdsong-recognition/train_audio/aldfly/XC2628.mp3"
#fp = "./../birdsong-recognition/train_audio/aldfly/XC16967.mp3"
# all have sampling rate 44100 or 48000 I think (perhaps irrelevant)

coef = mp3ToFreqDom(fp, 0.2)
plt.matshow(coef)
plt.show()


#x = np.arange(512)
#y=np.sin(2*np.pi*x/32)
#coef, freq = pywt.cwt(y, scales= np.arange(1,129), wavelet ='gaus1')
