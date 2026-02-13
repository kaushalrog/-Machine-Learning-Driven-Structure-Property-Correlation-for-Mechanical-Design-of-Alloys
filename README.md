# Machine Learning Driven Structure–Property Correlation for Alloy Phase Prediction

An AI-based system that predicts alloy phase stability using machine learning models such as Neural Networks, XGBoost, Random Forest, and SVM.

---

## Overview

This project uses machine learning to predict the phase structure of alloys based on their composition, thermodynamic properties, and processing parameters. Predicting alloy phases is critical in materials engineering because phase structure directly affects mechanical properties such as strength, hardness, and durability. Traditional experimental methods are expensive and time-consuming, so this project demonstrates how ML can accelerate alloy design and material discovery. 

The system analyzes features such as Valence Electron Concentration (VEC), entropy of mixing, enthalpy of mixing, atomic size difference, and synthesis parameters to classify alloys into phase categories like FCC, BCC, Intermetallic, and mixed phases.

---

## Features

* Predict alloy phase using Machine Learning
* Supports multiple ML models: Neural Network, XGBoost, Random Forest, and SVM
* Feature importance analysis to identify key physical parameters
* Data preprocessing including scaling, encoding, and cleaning
* High prediction accuracy with Neural Network achieving best performance

---

## Dataset

* Total Samples: 1420 alloy compositions
* Features: 16 input features including

  * Valence Electron Concentration (VEC)
  * Atomic Size Difference
  * Enthalpy of Mixing
  * Entropy of Mixing
  * Electronegativity Difference
  * Processing and synthesis parameters

Target Classes:

* FCC (Face Centered Cubic)
* BCC (Body Centered Cubic)
* Intermetallic (IM)
* Mixed phases

Dataset enables prediction of phase stability based on composition and structure. fileciteturn0file0

---

## Machine Learning Models Used

The following models were implemented and compared:

* Neural Network (MLPClassifier) → Best performance
* XGBoost → Strong generalization
* Random Forest → High accuracy and interpretability
* Support Vector Machine (SVM) → Moderate performance

Neural Network achieved highest test accuracy (~89%) and best balanced performance across all classes. fileciteturn0file0

---

## Workflow

1. Collect alloy composition and processing data
2. Clean and preprocess dataset
3. Encode categorical features
4. Scale numerical features
5. Train multiple ML models
6. Evaluate performance using accuracy and balanced accuracy
7. Compare models and select best performing model

Machine learning enables faster and cost-effective alloy design compared to traditional experimental approaches. fileciteturn0file1

---

## Tech Stack

* Python
* Scikit-learn
* XGBoost
* NumPy
* Pandas
* Matplotlib

---

## Results

Model Performance Summary:

* Neural Network → Best accuracy and generalization
* XGBoost → Strong and stable performance
* Random Forest → Reliable and interpretable
* SVM → Lowest performance among tested models

Key Features influencing prediction:

* Valence Electron Concentration (VEC)
* Enthalpy of Mixing
* Entropy of Mixing
* Atomic Size Difference

---

## Project Structure

```
project/
│── dataset/
│── models/
│── notebooks/
│── main.py
│── requirements.txt
│── README.md
```

---

## Applications

* Material science research
* Alloy design optimization
* Mechanical property prediction
* Accelerated material discovery

---

## Future Improvements

* Use Deep Learning architectures
* Increase dataset size
* Deploy as web application
* Integrate real-time material prediction

---

## Author

Kaushal S
AI and Robotics Developer
Amrita Vishwa Vidyapeetham

---

## License

MIT License
