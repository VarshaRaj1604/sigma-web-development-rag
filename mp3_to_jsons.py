import whisper
import os
import json

model = whisper.load_model("large-v2")

os.makedirs("jsons", exist_ok=True)

audios = os.listdir("audios")

for audio in audios:

    if "-" in audio:

        number = audio.split("-")[0]
        title = os.path.splitext(audio.split("-", 1)[1])[0]

        print(number, title)

        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False,
            fp16=False  # Recommended on CPU/Windows
        )

        chunks = []

        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }

        filename = os.path.splitext(audio)[0]

        with open(f"jsons/{filename}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)