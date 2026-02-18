# 🎵 Music Genre Classifier

Trabalho desenvolvido na disciplina de **Inteligência Artificial** — **UFCG — 2025.2**

---

## 👥 Contribuidores

* [Arthur](https://github.com/arthur-vasco7)
* [Carlos Artur](https://github.com/CarlosArturr)
* [Débora Pereira](https://github.com/DeboraSabrinaOPereira)
* [Rafael Cavalcante](https://github.com/rafaelmcavalcante)
* [Ruan Rodrigues](https://github.com/ruanrodrigues20)

---

## 🧪 Etapas do Projeto

1. Coleta dos dados de treinamento (PalcoMP3)
2. Criação dos testes com TDD
3. Extração de features das músicas
4. Implementação da MLP (Multi-Layer Perceptron)
5. Execução e validação dos testes

---

## 🎼 Features Extraídas

O pipeline de pré-processamento extrai as seguintes **features de áudio** usando `librosa`:

| Feature | O que Captura | Por que é útil |
|---------|---------------|----------------|
| **MFCC (mean/var)** | Timbre | Diferencia instrumentos e estilo de produção |
| **MFCC delta / delta²** | Mudança temporal do timbre | Captura movimentos e articulação do som |
| **Chroma STFT (mean/var)** | Notas / acordes | Identifica harmonia e progressão de acordes |
| **Spectral Centroid (mean/var)** | Brilho do som | Diferencia instrumentos ou gêneros mais agudos |
| **Spectral Bandwidth (mean/var)** | Largura espectral | Mede complexidade tonal |
| **Spectral Rolloff (mean/var)** | Frequência abaixo da qual X% da energia está | Destaca percussão e agudos |
| **Zero-Crossing Rate (ZCR)** | Número de vezes que o sinal cruza zero | Captura atividade ou ruído do som |
| **Root Mean Square (RMS) Energy** | Energia média | Mede volume e dinâmica |
| **Log-Mel Spectrogram** | Padrões de frequência perceptuais | Captura timbre e ritmo de forma detalhada |
| **Harmonic / Percussive RMS** | Energia separada em harmonia e percussão | Diferencia melodia e ritmo |
| **Spectral Contrast** | Diferença entre picos e vales do espectro | Captura riqueza tonal e dinâmica |
| **Tonnetz** | Relações harmônicas entre notas | Diferencia progressões de acordes por gênero |

> Essas features permitem que a MLP capture tanto o conteúdo harmônico e rítmico quanto o timbre e a textura do áudio, ajudando na classificação de gêneros musicais.

---

## ⚙️ Requisitos

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

## ▶️ Como Rodar

### 1️⃣ Ativar ambiente virtual

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 2️⃣ Pré-processar o dataset (somente se ainda não tiver feito)

```bash
python -m music_genre_classifier.data.preprocess
```

### 3️⃣ Treinar a MLP

```bash
python -m music_genre_classifier.main
```

### 4️⃣ Rodar a página web para testar

```bash
uvicorn music_genre_classifier.api.app:app --reload
```

Acesse no navegador:
👉 [http://localhost:8000/](http://localhost:8000/)
