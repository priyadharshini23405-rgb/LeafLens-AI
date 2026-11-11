# 🍃 LeafLens AI – Plant Disease Detection Using Deep Learning

**LeafLens AI** is an AI-powered plant disease detection system that uses deep learning and image classification to identify plant diseases from leaf images. It uses the **MobileNetV2** model to analyze uploaded leaf images and predict the possible disease with a confidence score.

The application is developed using Python, TensorFlow, and Streamlit, providing a simple and interactive interface for plant disease analysis.

## 🚀 Features

- **Leaf Disease Detection:** Upload a leaf image and predict its disease.
- **Confidence Score:** View the model's confidence percentage for each prediction.
- **Top 3 Predictions:** Display the three most likely disease classifications.
- **Disease Information:** Explore available plant disease classes, symptoms, and general guidance.
- **Prediction History:** View previous predictions during the current session.
- **CSV Export:** Download prediction history for further analysis.
- **Model Evaluation:** Evaluate the model using test images and view accuracy, classification reports, and a confusion matrix.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Deep learning model |
| MobileNetV2 | Image classification |
| Streamlit | Web application interface |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Matplotlib | Evaluation visualization |
| Scikit-learn | Model evaluation metrics |
| Pillow (PIL) | Image processing |

## 🧠 Model Architecture

LeafLens AI uses **MobileNetV2**, a pre-trained convolutional neural network architecture.

The model includes:

1. Image resizing to 160 × 160 pixels.
2. Data augmentation for improved training.
3. MobileNetV2 feature extraction.
4. Global Average Pooling.
5. Dropout layer to help reduce overfitting.
6. Dense softmax output layer for disease classification.

The model is trained to classify plant leaves across 38 disease and healthy-leaf categories.

## 📂 Project Structure

```text
LeafLens-AI/
│
├── app.py
├── train.py
├── test_model.py
├── requirements.txt
│
├── models/
│   ├── leaflens.keras
│   └── class_names.json
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/LeafLens-AI.git
```

### 2. Navigate to the Project Folder

```bash
cd LeafLens-AI
```

### 3. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 4. Activate the Virtual Environment

**For macOS / Linux:**

```bash
source .venv/bin/activate
```

**For Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 📊 Model Evaluation

The application includes a model evaluation module that displays:

- Overall test accuracy
- Correct and incorrect predictions
- Classification report
- Confusion matrix
- Individual image predictions

**Note:** Evaluation results depend on the available test images. The current test set is small and covers only a subset of the 38 classes. Therefore, its accuracy should not be treated as a complete measure of performance across all classes.

## 🌱 Applications

- Smart agriculture
- Early identification of plant diseases
- Plant health monitoring
- Agricultural technology research
- Supporting farmers with preliminary disease identification

## 🔮 Future Enhancements

- Add more real-world plant leaf images.
- Improve model accuracy through additional training and fine-tuning.
- Support more plant species and diseases.
- Integrate a plant disease information database.
- Deploy the application online for public access.
- Add multilingual support for farmers.

## ⚠️ Disclaimer

LeafLens AI is an educational project developed for plant disease classification. Its predictions and general guidance should not replace professional agricultural advice. Prediction accuracy may vary depending on image quality and the trained model.

## 👩‍💻 Project Information

**Project Name:** LeafLens AI  
**Domain:** Artificial Intelligence and Deep Learning  
**Project Type:** Mini Project  
**Model:** MobileNetV2  
**Interface:** Streamlit

---

⭐ If you find this project interesting, consider giving the repository a star!
