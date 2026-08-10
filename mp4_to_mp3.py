# Converts the videos to mp3 
from importlib.metadata import files
import os
import subprocess

for file in os.listdir("videos"):
    if file.endswith((".mp4", ".webm")):
        input_file = os.path.join("videos", file)
        output_file = os.path.join("audios", f"{os.path.splitext(file)[0]}.mp3")
        subprocess.run(["ffmpeg", "-i", input_file, "-q:a", "0", "-map", "a", output_file])