# ❤️ Cardiac Image Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Status](https://img.shields.io/badge/Project-Completed-green)

A deep learning project for **classifying cardiac medical images** using **Myocardial Perfusion Imaging (MPI)** scans.

The model automatically classifies heart scans into three clinical conditions:

* **Normal**
* **Ischemia**
* **Infarction**

The project was developed at the **University of Jordan** as part of a Machine Learning and Neural Networks course.

---

# 🧠 Project Overview

Cardiovascular diseases remain one of the leading causes of death worldwide.

Doctors commonly use **Myocardial Perfusion Imaging (MPI)** to examine blood flow in the heart muscle. However, manual interpretation of these images can be time-consuming and sometimes subjective.

This project explores how **Deep Learning** can assist in the diagnostic process by automatically analyzing MPI scans.

The system uses a **fine-tuned MobileNet CNN** model to classify cardiac images.

---

# ⚙️ Model Architecture

The model is based on **MobileNet (V1)** with transfer learning.

Architecture pipeline:

Input Image (224×224)
↓
Pretrained **MobileNet (ImageNet)**
↓
Global Average Pooling
↓
Dense Layer (128 neurons)
↓
Dropout (0.3)
↓
Dense Layer (256 neurons)
↓
Dropout (0.4)
↓
Softmax Output Layer (3 classes)

Only the **deeper layers were fine-tuned** to adapt the model to the MPI dataset.

---

# 📊 Dataset

The dataset contains **Myocardial Perfusion Imaging scans from 88 patients**.

Each patient includes three standard MPI views:

* **QPS AC**
* **Stress QGS AC**
* **Rest QGS AC**

Dataset distribution:

| Class      | Images |
| ---------- | ------ |
| Normal     | 189    |
| Ischemia   | 60     |
| Infarction | 12     |

Because the dataset was **imbalanced**, additional augmentation and synthetic samples were used to improve training.

---

# 🔧 Data Preprocessing

Before training the model, several preprocessing steps were applied:

* Removing incomplete patient cases
* Cropping images to remove personal information
* Resizing images to **224×224**
* Pixel normalization
* Dataset splitting

Dataset split:

* **Training:** ~55%
* **Validation:** ~20%
* **Testing:** ~25%

Data augmentation included:

* Small rotations
* Brightness variation
* Minor spatial shifts

---

# 📈 Training Configuration

| Parameter     | Value                     |
| ------------- | ------------------------- |
| Framework     | TensorFlow / Keras        |
| Optimizer     | Adam                      |
| Learning Rate | 0.0001                    |
| Epochs        | Up to 50                  |
| Batch Size    | 16                        |
| Loss Function | Categorical Cross Entropy |

Additional techniques used:

* **Dropout regularization**
* **Early stopping**
* **Class weighting for imbalance**

---

# 📊 Results

The trained model achieved:

**Test Accuracy: 91%**

Evaluation metrics included:

* Precision
* Recall
* F1-Score
* Confusion Matrix

The model achieved strong performance for **Normal and Infarction cases**, with slightly lower recall for **Ischemia** due to dataset imbalance.

---

# 🖥 GUI Application

A simple **Tkinter GUI** was developed to test the trained model.

Features:

* Upload MPI image
* Automatic preprocessing
* Load trained model
* Predict the class
* Display prediction confidence
* Show the uploaded image

This tool allows quick testing without running the full training pipeline.

---

# 🛠 Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* scikit-learn
* Matplotlib
* Seaborn
* Tkinter

---

# ⚠️ Challenges

### Limited Dataset

Only **88 patients** were available, which is relatively small for deep learning.

### Class Imbalance

The **Infarction class contained very few samples**, which required augmentation and class weighting.

### Visual Similarity

The visual difference between **Normal and mild Ischemia** cases is subtle, making classification difficult even for experienced cardiologists.

---

# 📌 Conclusion

This project demonstrates that **deep learning can assist in medical image classification**.

By fine-tuning a pretrained MobileNet model and applying proper preprocessing and regularization techniques, the model achieved strong performance despite the limited dataset.

Such systems may provide **valuable support tools for cardiologists**, helping with faster and more consistent diagnostic analysis.

---

# 👨‍💻 Authors

* Mohammad Hajjaj
* Ghayth Bani Yaseen
* Mohammad Mustafa
* Anas Nahas

University of Jordan
