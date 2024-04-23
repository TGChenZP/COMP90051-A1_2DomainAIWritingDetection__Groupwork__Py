## COMP90086 Computer Vision Group Project

#### Joint AI Generated Text detection from different sources 

Semester 1 2024, University of Melbourne

#### Group Members

- **Name:** Lang (Ron) Chen **Student ID:** 1181506 **Email:** Lachen1@student.unimelb.edu.au
- **Name:** Un Leng Kam **Student ID:** 1178863 **Email:** ukam@student.unimelb.edu.au
- **Name:** Di Wu **Student ID:** 1208784 **Email:** dww3@student.unimelb.edu.au

---

We built a system for AI generated text detection (NLP classification of human and AI written text) with source and inference data from two domains. Our primary solution is a Deep BERT based framework, with source code found in  `./PrivatePackages/pytorch` and Training UI found in `./main.py`. Features of our source code include Domain Adversarial Neural Network, manually defined loss functions as well as multiple architectures based on LSTM and Transformers.

Our experiments also consists non Deep Learning models as prototypes and proof of concepts (i.e. upsampling and auxillary features) which can be found in `./notebooks` alongside our data processing and auxillary feature generation notebooks. `./W2V_Pretraining.ipynb` is the training UI for pretraining based on W2V, of which results were not used in the final model.

`./PrivatePackages/Tuners` contain tuning packages used in non-DL experiments.

---

Please run `pip install -r requirements.txt` before running our code