import time
import librosa
import numpy as np

from music_genre_classifier.configs import get_logger, SAMPLE_RATE

logger = get_logger(__name__)


class FeatureExtractor:
    N_MFCC = 20

    @staticmethod
    def extract(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray | None:
        start = time.time()

        if y is None or len(y) < sr:
            logger.warning("⚠️ Áudio curto ou vazio")
            return None

        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=FeatureExtractor.N_MFCC)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            rms = librosa.feature.rms(y=y)

            def stats(x):
                return [x.mean(axis=1), x.var(axis=1)]

            features = np.hstack(
                [
                    *stats(mfcc),
                    *stats(mfcc_delta),
                    *stats(mfcc_delta2),
                    *stats(chroma),
                    *stats(centroid),
                    *stats(bandwidth),
                    *stats(rolloff),
                    *stats(zcr),
                    *stats(rms),
                ]
            )

            logger.info(f"Features extract in {time.time() - start:.2f}s")
            return features

        except Exception as e:
            logger.error(f"❌ Erro in extractor features: {e}")
            return None

    @staticmethod
    def get_feature_names():
        def stat_names(prefix, n):
            return [f"{prefix}_{i}_mean" for i in range(n)] + [
                f"{prefix}_{i}_var" for i in range(n)
            ]

        names = []
        names += stat_names("mfcc", 20)
        names += stat_names("mfcc_delta", 20)
        names += stat_names("mfcc_delta2", 20)
        names += stat_names("chroma", 12)

        names += [
            "spec_centroid_mean",
            "spec_centroid_var",
            "spec_bandwidth_mean",
            "spec_bandwidth_var",
            "spec_rolloff_mean",
            "spec_rolloff_var",
            "zcr_mean",
            "zcr_var",
            "rms_mean",
            "rms_var",
        ]

        return names
