import random
import shutil
import subprocess
from pathlib import Path


def check_dependencies():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("❌ ffmpeg not found. Please install it!")


class DatasetPreparer:
    def __init__(self, genre: str):
        self.genre = genre
        self.music_dir = Path(genre)
        self.test_dir = self.music_dir / "test"
        self.train_dir = self.music_dir / "train"

    def create_folders(self):
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)

    def shuffle_and_split(self):
        files = list(self.music_dir.glob("*.mp3"))
        random.shuffle(files)

        if len(files) < 200:
            raise RuntimeError("❌ Error: at least 200 .mp3 files are required")

        test = files[:50]
        train = files[50:200]

        for f in test:
            shutil.move(str(f), self.test_dir / f.name)
        for f in train:
            shutil.move(str(f), self.train_dir / f.name)

        print("✔️ Moved: 50 → test, 150 → train")

    def convert_to_wav(self):
        for dir_path in (self.test_dir, self.train_dir):
            for mp3 in dir_path.glob("*.mp3"):
                wav_path = mp3.with_suffix(".wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(mp3), str(wav_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                mp3.unlink()
        print("✔️ MP3 → WAV conversion completed")

    def rename_files(self):
        for dir_path in (self.test_dir, self.train_dir):
            files = list(dir_path.glob("*.wav"))
            random.shuffle(files)

            # Temporary step
            for f in files:
                f.rename(dir_path / f"__tmp__{f.name}")

            i = 0
            for f in dir_path.glob("__tmp__*.wav"):
                f.rename(dir_path / f"{self.genre}{i}.wav")
                i += 1

            print(f"✔️ Renamed in {dir_path}: {self.genre}0.wav ... {self.genre}{i - 1}.wav")


def run(genre: str):
    """
    Module entry point.
    """
    preparer = DatasetPreparer(genre)
    check_dependencies()
    preparer.create_folders()
    preparer.shuffle_and_split()
    preparer.convert_to_wav()
    preparer.rename_files()
    print(f"🎉 Process completed for '{genre}'")


if __name__ == "__main__":
    run("pop")
    run("classica")
    run("eletronica")
    run("forro")
    run("rock")
