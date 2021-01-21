import os
import argparse
import glob
import pywt
import numpy as np
import matplotlib.pyplot as plt
import pickle

from pathlib import Path
from pydub import AudioSegment
# import librosa

"""
Used for checking if paths exist in the argument parsing
"""
def dir_path(path):
	if os.path.isdir(path):
		return path
	else:
		raise argparse.ArgumentTypeError(f"readable_dir: {path} is not a valid path")

"""
Accepts a bird sound AudioSegment

Returns a list containing the sliced parts, needs more complexity as of yet
"""
def process_audio(input):
	output = []

	if len(input) >= 4000:
		n = len(input) // 4000

		for i in range(n):
			output.append(input[i*4000:3999+4000*i])

	return output



def main():
	print("Preprocessing of Kaggle bird audio data to either slices or raw audio waves")

	parser = argparse.ArgumentParser()
	parser.add_argument("bird_sounds_folder", type=dir_path, help="position of the kaggle bird sounds directory")
	parser.add_argument("target_folder", help="target directory of the processed bird sounds")
	parser.add_argument("-n", "--numberoffolders", type=int, default=-1, help="number of folders that are processed, counted from the start")
	parser.add_argument("-s", "--slice", help="slice all the audio fragments", action="store_true")
	parser.add_argument("-w", "--wave", help="transform all audio fragments into raw audio waves", action="store_true")
	args = parser.parse_args()

	if not (args.wave or args.slice):
		print("No audio handling option chosen")
		return

	print(f"Retrieving bird sounds from: {args.bird_sounds_folder}")

	origin_path = Path(args.bird_sounds_folder)
	destination_path = Path(args.target_folder)
	max_folders = args.numberoffolders

	try:
		destination_path.mkdir()
		print(f"Created directory for handled bird sounds at: {destination_path}")
	except FileExistsError:
		print(f"Putting handled bird sounds at: {destination_path}")

	print("Starting the audio processing.")
	print("Number of folders to be processed: "+ (str(max_folders) if max_folders != -1 else "all"))

	unprocessed_files = []
	process_counter = 0
	folder_counter = 0
	pickle_acc = []

	for entry in origin_path.iterdir():
		if entry.is_file():
			unprocessed_files.append(entry.name)

		if entry.is_dir():
			if max_folders != -1 and folder_counter >= max_folders:
				print("Requested number of folders processed reached")
				break
			folder_counter += 1

			pickle_acc.append([entry.name, []])

			sub_dest = destination_path / entry.name
			sub_dest.mkdir(exist_ok=True)

			for filename in entry.glob("*.mp3"):
				process_counter += 1

				audio = AudioSegment.from_file(filename)

				if args.slice:
					audio_fragments = process_audio(audio)

					for i in range(len(audio_fragments)):
						audio_fragments[i].export(sub_dest / (filename.stem + "_" + str(i) + ".wav"), format="wav")


################################################################################
#
#	Hier kan je dus doen wat je wilde doen met die data transformatie
#	All these numbers are up for debate
#
################################################################################
				if args.wave:
					audio = audio.set_frame_rate(int(audio.frame_rate / 100))
					
					# For returning the crappy frame_rate audio
					# print(audio.frame_rate)
					# audio.export(sub_dest / (filename.stem + ".wav"), format="wav")
					
					channel_sounds = audio.split_to_mono()
					sample = audio.get_array_of_samples()
					audio_arr = np.array(sample).T.astype(np.float32)
					audio_arr /= np.iinfo(sample.typecode).max
					coef, freq = pywt.cwt(audio_arr, scales=np.arange(1, 29), wavelet='gaus1')

					pickle_acc[-1][1].append(coef)

					# For looking at the pretty pictures
					# plt.matshow(coef[:,1:28])
					# plt.show()

				print("Processing folder:", entry.name, " File number:", process_counter, end='\r')

	if unprocessed_files:
		print("If you also want these files processed, put them in a subdirectory.")
	for filename in unprocessed_files:
		print(f"\t{filename}")

	if args.wave and pickle_acc:
		with open(destination_path / "bird_sound_data.pkl", "wb") as f:
			pickle.dump(pickle_acc, f)

if __name__ == "__main__":
	main()