const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const form = document.getElementById("uploadForm");
const result = document.getElementById("result");

uploadBox.addEventListener("click", () => fileInput.click());

uploadBox.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadBox.classList.add("dragover");
});

uploadBox.addEventListener("dragleave", () => {
    uploadBox.classList.remove("dragover");
});

uploadBox.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadBox.classList.remove("dragover");
    fileInput.files = e.dataTransfer.files;
    showFileName();
});

fileInput.addEventListener("change", showFileName);

function showFileName() {
    if (fileInput.files.length > 0) {
        fileName.textContent = "Arquivo selecionado: " + fileInput.files[0].name;
    }
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!fileInput.files.length) return;

    result.textContent = "Processando...";

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        result.textContent = "Resultado: " + data.genre.toUpperCase();

    } catch (error) {
        result.textContent = "Erro ao processar.";
    }
});