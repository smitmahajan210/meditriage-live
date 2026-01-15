# 🏥 MediTriage: AI-Powered Hospital Assistant

[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://meditriage-live.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

**MediTriage** is a full-stack healthcare application designed to streamline patient intake and prioritize care using Artificial Intelligence. It bridges the gap between raw medical data and actionable clinical insights.

🔗 **Live Demo:** [https://meditriage-live.onrender.com](https://meditriage-live.onrender.com)  
*(Note: Please allow ~50 seconds for the free tier server to wake up on first load.)*

---

## 🚀 Key Features

### 1. 📝 Handwriting Digitization (OCR)
* **Problem:** Manual data entry from handwritten patient forms is slow and error-prone.
* **Solution:** Automatically extracts Patient IDs from handwritten images.
* **Tech:** Implemented using **Convolutional Neural Networks (CNN)** with the **VGG13** architecture for high-accuracy character recognition.

### 2. ⚡ Patient Risk Assessment (Triage)
* **Problem:** Emergency rooms need to instantly prioritize patients based on vitals.
* **Solution:** Predicts health risk levels (Critical, Urgent, Non-Urgent) based on 7 key bio-markers (Glucose, Insulin, BMI, Age, etc.).
* **Tech:** Powered by a **Multi-Layer Perceptron (MLP)** Neural Network trained on clinical datasets.

### 3. 🖥️ Real-Time Dashboard
* **Interface:** Interactive UI built with **Streamlit** for seamless user experience.
* **Backend:** High-performance **FastAPI** backend handling asynchronous inference requests.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Interactive web interface for data entry and visualization. |
| **Backend** | FastAPI | High-speed API for serving model predictions. |
| **Deep Learning** | PyTorch / OpenAI | CNNs for image processing and MLPs for risk prediction. |
| **Data Processing** | NumPy & Pandas | Data manipulation and preprocessing pipelines. |
| **Deployment** | Render | Containerized cloud deployment with automatic CI/CD. |
| **Version Control** | Git & GitHub | Source code management. |

---

## 📸 Screenshots

### Risk Assessment Dashboard
*(Add your screenshot `image_41a42d.png` here)*
`![Dashboard Screenshot](path/to/your/image.png)`

---

## 🏗️ Architecture

```mermaid
graph LR
    A[User (Streamlit UI)] -- Sends Vitals/Image --> B[FastAPI Backend]
    B -- Process Image --> C[CNN / VGG13 Model]
    B -- Analyze Vitals --> D[MLP / Risk Model]
    C --> E[Prediction Result]
    D --> E
    E -- JSON Response --> A
