import os
import argparse
import glob
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
	print("Preprocessing of Kaggle bird audio data to 4 second windows of bird sounds, removing long silences")

	parser = argparse.ArgumentParser()
	parser.add_argument("bird_sounds_folder", type=dir_path, help="position of the kaggle bird sounds directory")
	parser.add_argument("target_folder", help="target directory of the processed bird sounds")
	args = parser.parse_args()

	print(f"Retrieving bird sounds from: {args.bird_sounds_folder}")

	origin_path = Path(args.bird_sounds_folder)
	destination_path = Path(args.target_folder)

	try:
		destination_path.mkdir()
		print(f"Created directory for sliced bird sounds at: {args.target_folder}")
	except FileExistsError:
		print(f"Putting sliced bird sounds at: {args.target_folder}")

	print("Starting the slicing")

	unprocessed_files = []
	process_counter = 0

	for entry in origin_path.iterdir():
		if entry.is_file():
			unprocessed_files.append(entry.name)

		if entry.is_dir():

			sub_dest = destination_path / entry.name
			sub_dest.mkdir(exist_ok=True)

			for filename in entry.glob("*.mp3"):
				process_counter += 1

				audio = AudioSegment.from_file(filename)
				audio_fragments = process_audio(audio)

				for i in range(len(audio_fragments)):
					audio_fragments[i].export(sub_dest / (filename.stem + "_" + str(i) + ".wav"), format="wav")


				print("Processing folder:", entry.name, " File number:", process_counter, end='\r')

	if unprocessed_files:
		print("If you also want these files processed, put them in a subdirectory")
	for filename in unprocessed_files:
		print(f"\t{filename}")


if __name__ == "__main__":
	main()