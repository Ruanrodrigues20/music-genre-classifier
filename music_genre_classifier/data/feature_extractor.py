import time
import librosa
import numpy as np

from music_genre_classifier.configs import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    @staticmethod
    def extract(y: np.ndarray, sr: int) -> np.ndarray | None:
        start = time.time()

        if y is None or len(y) < sr:
            logger.warning("⚠️ Áudio curto ou vazio")
            return None

        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            rms = librosa.feature.rms(y=y)

            features = np.hstack(
                [
                    mfcc.mean(axis=1),
                    mfcc.var(axis=1),
                    mfcc_delta.mean(axis=1),
                    mfcc_delta.var(axis=1),
                    mfcc_delta2.mean(axis=1),
                    mfcc_delta2.var(axis=1),
                    chroma.mean(axis=1),
                    chroma.var(axis=1),
                    spec_centroid.mean(axis=1),
                    spec_centroid.var(axis=1),
                    spec_bandwidth.mean(axis=1),
                    spec_bandwidth.var(axis=1),
                    spec_rolloff.mean(axis=1),
                    spec_rolloff.var(axis=1),
                    zcr.mean(axis=1),
                    zcr.var(axis=1),
                    rms.mean(axis=1),
                    rms.var(axis=1),
                ]
            )

            logger.info(f"Features extract in {time.time() - start:.2f}s")
            return features

        except Exception as e:
            logger.error(f"❌ Erro in extractor features: {e}")
            return None
