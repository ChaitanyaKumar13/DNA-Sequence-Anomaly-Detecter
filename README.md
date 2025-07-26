# 🧬 DNA Sequence Anomaly Detection with Transformers

This project builds a **Transformer-based deep learning model** to classify DNA sequences and **flag uncertain predictions as anomalies**. It also provides an **interactive Streamlit web app** where users can upload DNA data, view predictions, and download logs.

---

## 🚀 Features

- 🔬 **DNA Sequence Classifier** using a Transformer model
- ⚠️ **Anomaly Detection** via Monte Carlo Dropout + Uncertainty Estimation
- 📊 Confidence & Uncertainty scores for each prediction
- 🖥️ Streamlit Web App with:
  - File upload
  - Prediction table
  - Red-highlighted anomalies
  - CSV export of results

---

## 📁 Folder Structure

.
├── app.py # Streamlit App
├── transformer_dna_classifier.py # Transformer model definition
├── utils/
│ └── load_transformer_data.py # Preprocessing & encoding utils
├── models/
│ └── transformer_dna_uncertain.keras # Trained model
├── data/
│ └── evaluation_set.csv # Evaluation DNA sequences
├── dna_predictions.csv # Exported predictions
└── requirements.txt # Python dependencies

## 🧪 Model Details

- **Model Type:** Transformer with Positional Encoding
- **Input:** One-hot encoded DNA sequence of length 180
- **Output:** Class probabilities (EI / IE / Neither)
- **Uncertainty Detection:** MC Dropout (30 iterations)
- **Anomaly Rule:** Confidence < `0.55` is flagged as ⚠️ Anomaly

---

## 📊 Classes

| Code    | Description     |
|---------|-----------------|
| `EI`    | Exon → Intron   |
| `IE`    | Intron → Exon   |
| `Neither` | Neither boundary |

---

## 💻 How to Run the App

🔧 Step 1: Install Requirements
pip install -r requirements.txt

▶️ Step 2: Start Streamlit App
streamlit run app.py

📂 Step 3: Upload DNA CSV File
The CSV should have 180 columns of binary (0/1) one-hot encoded data and a class column.

Example file: data/evaluation_set.csv

📌 Output
🧾 Table with:

True label

Predicted label

Confidence

Uncertainty

Status: Normal or ⚠️ Anomaly

✅ Anomalies highlighted in red

⬇️ Export results with a single click

📥 Example Input Format
csv
Copy
Edit
A1,A2,...,A180,class
0,1,...,0,EI
1,0,...,1,Neither
...
📤 Output Example
True Label	Predicted Label	Confidence	Uncertainty	Status
EI	Neither	0.545	0.020	⚠️ Anomaly
Neither	Neither	0.600	0.010	Normal

📚 Technologies Used
Python 🐍

TensorFlow / Keras 🤖

NumPy / Pandas / scikit-learn

Streamlit 🌐

Matplotlib / Seaborn (for optional visualization)

🛠️ Future Improvements
Add support for DNA sequence input (A, C, G, T) instead of just one-hot vectors

Include BERT-based DNA model for better accuracy

Visualize positional attention weights

📃 License
This project is for educational purposes and research prototyping only.

🙋‍♂️ Author
Chaitanya Kumar
LinkedIn: https://www.linkedin.com/in/chaitanya-kumar-78a4b524b/ 
GitHub: https://github.com/ChaitanyaKumar13
