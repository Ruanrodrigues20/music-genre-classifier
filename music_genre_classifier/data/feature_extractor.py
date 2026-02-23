import time
import librosa
import numpy as np

from music_genre_classifier.configs import get_logger, SAMPLE_RATE

logger = get_logger(__name__)


class FeatureExtractor:
    N_MFCC = 20
    N_MELS = 40
    EPS = 1e-8

    @staticmethod
    def _stats(x: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(x, dtype=np.float32)
        return [x.mean(axis=1), x.var(axis=1)]

    @staticmethod
    def extract(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray | None:
        start = time.time()

        if y is None or len(y) < sr:
            logger.warning("⚠️ Áudio curto ou vazio")
            return None

        try:
            y = np.asarray(y, dtype=np.float32)

            y = librosa.util.normalize(y)

            stats = FeatureExtractor._stats

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=FeatureExtractor.N_MFCC)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)

            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            rms = librosa.feature.rms(y=y)

            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=FeatureExtractor.N_MELS
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)

            tempo = librosa.feature.tempo(y=y, sr=sr)
            tempo_bpm = float(tempo[0]) if tempo is not None and len(tempo) > 0 else 0.0

            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_env_2d = onset_env.reshape(1, -1)

            max_size = min(200, len(onset_env))
            if max_size > 2:
                ac = librosa.autocorrelate(onset_env, max_size=max_size)
                pulse_clarity = (
                    float(np.max(ac[1:]) / (np.mean(ac[1:]) + FeatureExtractor.EPS))
                    if ac.size > 2
                    else 0.0
                )
            else:
                pulse_clarity = 0.0

            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            flatness = librosa.feature.spectral_flatness(y=y)

            y_harm = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)

            peak = float(np.max(np.abs(y))) + FeatureExtractor.EPS
            rms_scalar = float(np.sqrt(np.mean(y**2))) + FeatureExtractor.EPS
            crest_factor = float(peak / rms_scalar)

            rms_delta = np.diff(rms, axis=1)
            rms_delta = (
                rms_delta if rms_delta.size > 0 else np.zeros((1, 1), dtype=np.float32)
            )

            features = np.hstack(
                [
                    *stats(mfcc),
                    *stats(mfcc_delta),
                    *stats(mfcc_delta2),
                    *stats(chroma),
                    *stats(chroma_cqt),
                    *stats(centroid),
                    *stats(bandwidth),
                    *stats(rolloff),
                    *stats(zcr),
                    *stats(rms),
                    *stats(log_mel),
                    np.array([tempo_bpm], dtype=np.float32),
                    *stats(onset_env_2d),
                    np.array([pulse_clarity], dtype=np.float32),
                    *stats(spec_contrast),
                    *stats(flatness),
                    *stats(tonnetz),
                    np.array([crest_factor], dtype=np.float32),
                    *stats(rms_delta),
                ]
            ).astype(np.float32)

            logger.info(f"Features extract in {time.time() - start:.2f}s")
            return features

        except Exception as e:
            logger.error(f"❌ Erro in extractor features: {e}")
            return None

    @staticmethod
    def get_feature_names() -> list[str]:
        def stat_names(prefix: str, n: int) -> list[str]:
            return [f"{prefix}_{i}_mean" for i in range(n)] + [
                f"{prefix}_{i}_var" for i in range(n)
            ]

        names: list[str] = []

        names += stat_names("mfcc", 20)
        names += stat_names("mfcc_delta", 20)
        names += stat_names("mfcc_delta2", 20)

        names += stat_names("chroma", 12)
        names += stat_names("chroma_cqt", 12)

        names += ["spec_centroid_mean", "spec_centroid_var"]
        names += ["spec_bandwidth_mean", "spec_bandwidth_var"]
        names += ["spec_rolloff_mean", "spec_rolloff_var"]
        names += ["zcr_mean", "zcr_var"]
        names += ["rms_mean", "rms_var"]

        names += stat_names("log_mel", 40)

        names += ["tempo_bpm"]
        names += ["onset_strength_mean", "onset_strength_var"]
        names += ["pulse_clarity"]

        names += stat_names("spec_contrast", 7)
        names += ["spec_flatness_mean", "spec_flatness_var"]

        names += stat_names("tonnetz", 6)

        names += ["crest_factor"]
        names += ["rms_delta_mean", "rms_delta_var"]

        return names
