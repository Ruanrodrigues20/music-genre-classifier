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

* Coleta dos dados de treinamento (PalcoMP3)
* Criação dos testes com TDD
* Extração de features das músicas
* Implementação da MLP
* Execução e validação dos testes

---

## ⚙️ Requisitos

* `ffmpeg` instalado
    
    ```bash
    sudo apt install ffmpeg -y # Base Debian
    sudo dnf install ffmpeg -y # Base Fedora
    ```

* `uv` instalado

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

* Python **3.12.x**
* Dataset disponível (caso a MLP ainda não tenha sido treinada)

---

## ▶️ Como Rodar

### **1. Setar enviroment:**
```bash
uv venv
source .venv/bin/activate
uv sync
cd src
```

### **2. Padroniza dataset:**
```bash
python -m music_genre_classifier.data.preprocess
```

### **3. Treinamento:**
```bash
python -m music_genre_classifier.main
```

### **4. Run pagina Web com Resultado e teste:**
```bash
uvicorn src.music_genre_classifier.api.app:app
```

Acesse no navegador:
👉 [http://localhost:8000/](http://localhost:8000/)
