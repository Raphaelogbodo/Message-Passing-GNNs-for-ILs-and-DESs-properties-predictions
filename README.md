# Graph-Neural-Network-for-Ionic-Liquids-and-Deep-Eutectics-Solvent-Property-Prediction
A Deep Learning Framework for Predicting Physicochemical Properties of Ionic Liquids and Deep Eutectic Solvents using Message Passing Neural Networks (NNConv)

## 🌍 Overview

This project focuses on developing a **Graph Neural Network (GNN)** model based on the **Neural Network Convolution (NNConv)** architecture for the **prediction of key physicochemical properties of Ionic Liquids (ILs)**.  

It represents an intersection of **computational chemistry**, **cheminformatics**, and **deep learning**, leveraging molecular graph representations and message passing to capture intricate structural and electronic interactions within IL systems.

---

## ⚙️ Motivation

Ionic liquids (ILs) are versatile materials with broad applications in **green chemistry**, **energy storage**, **lubrication**, and **separation processes**. However, their vast chemical space makes experimental property measurement infeasible for all combinations.

This project aims to address that challenge by developing a **data-driven predictive framework** that learns directly from molecular structures to estimate thermophysical and transport properties.

---

## 🧩 Key Features

- **NNConv-based GNN Architecture:**  
  Implements a flexible **message passing neural network** that dynamically learns edge-conditioned transformations between atom pairs.

- **Transfer Learning Extension:**  
  The trained IL model can be fine-tuned for **Deep Eutectic Solvents (DESs)**, demonstrating model generalization across related solvent families.

- **Comprehensive Cheminformatics Integration:**  
  Utilizes tools such as **RDKit** and **OpenBabel** for:
  - Molecular graph construction  
  - Descriptor and fingerprint generation  
  - SMILES parsing and molecular standardization  

- **Data Pipeline and Preprocessing:**  
  - Accesses and preprocesses IL property data directly from the **NIST ILThermo database**.  
  - Includes data cleaning, feature normalization, and dataset splitting utilities.  
  - Handles both **cation-anion pair representation** and global system descriptors.

- **End-to-End Reproducibility:**  
  Provides all scripts and utilities to:
  - Train the model on new datasets  
  - Evaluate model performance  
  - Visualize learning curves and prediction accuracy  

---

## 🧠 Model Architecture

The core architecture uses the **NNConv** layer from PyTorch Geometric to learn edge-conditioned message functions, enabling adaptive feature propagation across molecular graphs.

**Highlights:**
- Node embeddings represent atom-level features (atomic number, degree, valence, etc.)  
- Edge embeddings represent bond information (type, conjugation, ring status, etc.)  
- Global features incorporate molecular-level descriptors (e.g., polarity, volume, density)  
- Final readout combines local and global embeddings through fully connected layers

---

## 📚 Technologies Used

| Category | Tools/Libraries |
|-----------|-----------------|
| **Deep Learning Framework** | PyTorch, PyTorch Geometric |
| **Cheminformatics** | RDKit, OpenBabel |
| **Data Handling** | pandas, numpy, scikit-learn, Scipy |
| **Visualization** | matplotlib, seaborn |
| **Database Access** | NIST ILThermo data integration |

---

## 🧮 Datasets

- **Primary Source:** [NIST ILThermo Database](https://ilthermo.boulder.nist.gov/)
- **Properties Modeled:** e.g., density, viscosity, conductivity, etc.
- **Data Processing:** includes duplicate removal, removing outliers, removing undefined liquids, and molecular pairing, etc..

---
