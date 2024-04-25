## COMP90051 SML Assignment 1

#### Joint AI Generated Text detection from different sources 

Semester 1 2024, University of Melbourne

#### Group Members

- **Name:** Lang (Ron) Chen **Student ID:** 1181506 **Email:** Lachen1@student.unimelb.edu.au
- **Name:** Un Leng Kam **Student ID:** 1178863 **Email:** ukam@student.unimelb.edu.au
- **Name:** Di Wu **Student ID:** 1208784 **Email:** dww3@student.unimelb.edu.au

---

We built a system for AI generated text detection (NLP classification of human and AI written text) with source and inference data from two domains. Our primary solution is a Deep BERT based framework, but also conducted non-Deep Learning modelsas prototypes and proof of concepts. Below are a list of our key notebooks/directories and their functionalities

- `./PrivatePackages/pytorch` contain data factory, PyTorch Deep model (BERT and LSTM and W2V) definitions. Domain Adversarial Nerual Network and self-defined losses can also be found here.

- `./main.py` and `./W2V_Pretraining.ipynb` is the training UI for deep learning models and pretraining. Please run the scripts of `./notebooks` to get the engineered features before running deep training.

- `./notebooks` contain non-deep experiments, EDA scripts and feature/data engineering notebooks. Of particular importance is `1_Anderson_SeperateDomainCLF_experiment.ipynb` which created the classifier predictor for 'domain'. Please run in order

- `./notebooks/Tuning` contain tuning scrips for non-Deep models.

- `./PrivatePackages/Tuners` contain tuning packages used in non-DL experiments.

---

Please run `pip install -r requirements.txt` before running our code