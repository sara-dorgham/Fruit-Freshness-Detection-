# Fruit-Freshness-Detection-
A multi-task deep learning model that predicts fruit freshness from images by performing two tasks: 1) Classification: Fresh vs Rotten 2) Regression: Days remaining until spoilage Using a fine-tuned ResNet50, the system processes the dataset, trains both tasks, and outputs evaluation metrics and a final trained model.

 1) Dataset Preparation(Sara Dorgham)
- Images are loaded from `train / test / valid` folders.  
- Renamed and saved inside a unified folder: `augmented_dataset/`  
- A `labels.csv` file is generated containing:
  - image path  
  - days_to_rot  
  - original split  
  - class (Fresh/Rotten)  
- Extra augmentation is applied for Fresh class in training.

  2) Data Loading(Malak Khaled)
A custom PyTorch Dataset is implemented with:
- Resize  
- Augmentation (Rotation, ColorJitter, Affine…)  
- Normalization  
DataLoaders are created for:
- train  
- validation  
- test

  2) Data Analysis (EDA)(Eman Kilany)
The notebook generates:
- Distribution of classes  
- Distribution of days_to_rot  
- Train/Valid/Test statistics  
- Sample image visualizations  
Outputs are saved automatically (PNG files).

 4) Model Architecture(George)
 **Base:** ResNet50 (pretrained)

 Added Heads:
**Regression Head:**  
- Dense(512→256→128→1) + ReLU + BatchNorm + Dropout

**Classification Head:**  
- Dense(256→128→2) + ReLU + Dropout

 5) Training Strategy(Nada Nabil)
- Loss = (0.7 × MSELoss) + (0.3 × CrossEntropyLoss)  
- Optimizer: Adam  
- Scheduler: ReduceLROnPlateau  
- Early stopping  
- Fine-tuning last layers of ResNet  
- Saving best model weights  
- Tracking full training history (loss + accuracy)

   6) Evaluation(Menna Elasaly)
Model evaluation includes:
- MAE, MSE, RMSE, R²  
- Classification accuracy  
- Confusion matrix  
- Classification report  
- Visualization of predictions

7) Graphical User Interface (GUI) (Youssef Ahmed)
Graphical User Interface was developed to visualize the true freshness labels of any image in the dataset.
The GUI provides a clean, interactive interface with two main panels: image display and freshness results panel.

   8) How to Run
1. Clone the repository  
2. Place dataset folders inside main directory  
3. Open and run the notebook  
4. The model will be trained and evaluation results will be saved

9)Dataset Download:
https://drive.google.com/file/d/1K5ZLpqkjQplChZj7egE-r3t1DJ42sE7C/view?usp=sharing
