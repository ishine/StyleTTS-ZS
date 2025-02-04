from phonemizer import phonemize
import os
from tqdm import tqdm

def get_audio_files(path):
    """Get all audio files in the list."""
    audio_files = []
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            audio_files.append(line.strip())
    return audio_files

cv_path = "../Corpora/CommonVoice18"

phonemized_lines = []
for lang in os.listdir(cv_path):
    with open(os.path.join(cv_path,lang,"train.tsv"), "r") as f:  # Path to output.txt
        lines = f.readlines()

    # Phonemize the transcriptions
    phonemized = []
    filenames = []
    transcriptions = []
    speakers = []
    root_path = os.path.join(cv_path, lang, "clips")
    audio_files = get_audio_files(f"Data/audio_files_{lang}.txt")
    for (
        line
    ) in lines[1:]:  # Split filenames, text and speaker without phonemizing. Prevents mem error
        speaker, filename, _, transcription, _,_,_,_,_,_,_,_ = line.strip().split("\t")
        #speaker = filename.split('.')[0].split('_')[-1]
        if filename in audio_files:
            filenames.append(os.path.join(root_path, filename.replace('mp3','wav')))
            transcriptions.append(transcription)
            speakers.append(speaker)
    
    if lang=="en":
        lang="en-us"
    if lang=="es":
        lang="es"
    if lang=="fr":
        lang="fr-fr"

    # Phonemize all text in one go to avoid triggering  mem error
    phonemized = phonemize(
        transcriptions,
        language=lang,
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
        language_switch='remove-flags'
    )

    for i in tqdm(range(len(filenames))):
        phonemized_lines.append(
            (filenames[i], f"{filenames[i]}||{phonemized[i]}|{speakers[i]}|\n")
        )
    f.close()


phonemized_lines.sort(key=lambda x: int(x[0].split("_")[-1].split(".")[0]))

# Split training/validation set
train_lines = phonemized_lines[: int(len(phonemized_lines) * 0.9)]
val_lines = phonemized_lines[int(len(phonemized_lines) * 0.9) :]

with open(
        "Data/train_list.txt", "w+", encoding="utf-8"
    ) as f:  # Path for train_list.txt in the training data folder
        for _, line in train_lines:
            f.write(line)

with open(
        "Data/val_list.txt", "w+", encoding="utf-8"
    ) as f:  # Path for val_list.txt in the training data folder
        for _, line in val_lines:
            f.write(line)