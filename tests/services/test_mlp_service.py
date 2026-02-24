from music_genre_classifier.services import mlp_service

def test_exemplo():
    mlp = mlp_service.MlpService()
    mlp.preprocess()
    mlp.train()

