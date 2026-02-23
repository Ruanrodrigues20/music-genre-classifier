# Music Genre Classifier

Projeto desenvolvido na disciplina de **Inteligência Artificial** — **UFCG (2025.2)**.  
O sistema realiza **classificação automática de gêneros musicais** a partir de áudios, com pipeline completo: construção de dataset, pré-processamento, extração de features e treinamento.

---

## Contribuidores

- [Arthur](https://github.com/arthur-vasco7)
- [Carlos Artur](https://github.com/CarlosArturr)
- [Débora Pereira](https://github.com/DeboraSabrinaOPereira)
- [Rafael Cavalcante](https://github.com/rafaelmcavalcante)
- [Ruan Rodrigues](https://github.com/ruanrodrigues20)

---

## Objetivo

Construir um **classificador de gêneros musicais** com foco em:

- engenharia de features acústicas com **Librosa**;
- treinamento de uma **MLP (Multi-Layer Perceptron)** com **Scikit-learn**;
- avaliação por métricas multiclasses: **Accuracy, Precision, Recall, F1-score e Macro F1**;
- integração com **API FastAPI** e interface web para inferência.

Gêneros suportados: **POP, ROCK, ELETRÔNICA, CLÁSSICA e FORRÓ**.

---

## Visão Geral do Pipeline

1. **Coleta e organização** das músicas (dataset próprio)
2. **Pré-processamento**
   - conversão/segmentação e padronização de duração (5 segmentos de 6s → ~30s)
   - geração de `.wav` e renomeação por classe
3. **Extração de features** (vetor fixo por música)
4. **Treinamento da MLP** com normalização (`StandardScaler`)
5. **Avaliação + geração de gráficos**
6. **Inferência via Web/API** (upload de áudio → retorno do gênero)

---

## Pré-processamento do Dataset

O pré-processamento é feito no módulo `music_genre_classifier.utils.Preprocess`:

- Carrega arquivos `.mp3` por classe em `dataset/<genero>/`
- Seleciona 5 trechos de 6 segundos ao longo da música (total ~30s)
- Salva um novo arquivo `_30s.wav` e renomeia como `<genero><n>.wav`

Parâmetros principais:

- `SAMPLE_RATE = 22050`
- `SEGMENT_DURATION = 6`
- `NUM_SEGMENTS = 5`

---

## Extração de Features

A extração de features é determinística e gera vetores de dimensão fixa.  
As features são agregadas por **média e variância** (quando aplicável), garantindo consistência entre amostras.

### Normalização
O áudio é normalizado com `librosa.util.normalize(y)` para reduzir variações de volume/energia entre músicas.

### Features utilizadas

| Categoria | Features | Observação |
|---------|----------|------------|
| Timbre | MFCC (20) + Δ + Δ² | Base forte para caracterizar timbre e variação temporal |
| Harmonia | Chroma STFT (12), Chroma CQT (12) | Caracteriza tonalidade e acordes |
| Espectrais (básicas) | Centroid, Bandwidth, Rolloff, ZCR, RMS | Brilho, energia em altas frequências, percussividade |
| Percepção humana | Log-Mel (40 bandas) | Captura energia por bandas perceptuais |
| Ritmo/tempo | Tempo (BPM), Onset Strength, Pulse Clarity | Características de batida e pulso |
| Textura/instrumentação | Spectral Contrast (7), Spectral Flatness | Distingue sons tonais vs ruidosos, contraste espectral |
| Harmonia avançada | Tonnetz (6) | Relações harmônicas |
| Dinâmica | Crest Factor, RMS Delta | Pico vs energia média e variação de energia |

> A ordem e os nomes das colunas são mantidos em `FeatureExtractor.get_feature_names()` e são usados na geração do CSV.

---

## Modelo de Machine Learning

Treinamento usando **MLPClassifier** (Scikit-learn), com `StandardScaler` aplicado às features.

Configuração atual (mantida consistente entre `MLPConfig` e `model_config.json`):

```json
{
  "hidden_layer_sizes": [128, 64],
  "activation": "relu",
  "solver": "adam",
  "max_iter": 800,
  "alpha": 0.02,
  "learning_rate_init": 0.001,
  "early_stopping": true,
  "validation_fraction": 0.1,
  "n_iter_no_change": 20,
  "verbose": true,
  "random_state": 42,
  "batch_size": 32
}
```

## Requisitos

* **ffmpeg**

```bash
# Debian / Ubuntu
sudo apt install ffmpeg -y

# Fedora
sudo dnf install ffmpeg -y
````

* **uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

* **Python 3.12.x**
* Dataset disponível (caso a MLP ainda não tenha sido treinada), ou usar os CSV de test e train, feitos para adiantar
essa extração.

---

## Como Rodar

### 1 - Ativar ambiente virtual

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 2 - Pré-processar o dataset (somente se ainda não tiver feito)

```bash
python -m music_genre_classifier.data.preprocess
```

### 3 - Treinar a MLP

```bash
python -m music_genre_classifier.main
```

### 4 -  Rodar a página web para testar

```bash
uvicorn music_genre_classifier.api.app:app --reload
```

Acesse no navegador:
👉 [http://localhost:8000/](http://localhost:8000/)
