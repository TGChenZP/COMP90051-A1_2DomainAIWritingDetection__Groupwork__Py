# COMP90051-A1__Groupwork__Py
Authors: `Lang (Ron) Chen`, `Un Leng Kam`, `Di Wu`

Semester 1 2024, University of Melbourne

---

We built a system for AI generated text detection (NLP classification of human and AI written text) with source and inference data from two domains. Our primary solution is a Deep BERT based framework, with source code found in  `./PrivatePackages/pytorch` and Training UI found in `./main.py`. Features of our source code include Domain Adversarial Neural Network, manually defined loss functions as well as multiple architectures based on LSTM and Transformers.

Our experiments also consists non Deep Learning models as prototypes and proof of concepts (i.e. upsampling and auxillary features) which can be found in `./notebooks` alongside our data processing and auxillary feature generation notebooks. `./W2V_Pretraining.ipynb` is the training UI for pretraining based on W2V, of which results were not used in the final model.

`./PrivatePackages/Tuners` contain tuning packages used in non-DL experiments.

---

Please run `pip install -r requirements.txt` before running our code