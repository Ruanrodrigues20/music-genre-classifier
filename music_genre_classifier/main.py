from music_genre_classifier.services import MlpService


if __name__ == "__main__":
    mlp = MlpService()
    mlp.preprocess()
    mlp.train()
