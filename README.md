# Insurance Charges Prediction using Neural Networks

This project implements a deep learning model to predict insurance charges based on various patient characteristics using an Artificial Neural Network (ANN).

## 📊 Dataset Overview

The project uses the **Insurance Dataset** (`insurance.csv`) containing the following features:

- **age**: Age of the primary beneficiary
- **sex**: Gender of the insurance contractor (male/female)
- **bmi**: Body mass index
- **children**: Number of children covered by health insurance
- **smoker**: Smoking status (yes/no)
- **region**: Residential area in the US (northeast, northwest, southeast, southwest)
- **charges**: Individual medical costs billed by health insurance (target variable)

## 🛠️ Workflow

### 1. Data Loading and Exploration
- Loaded the insurance dataset using pandas
- Examined dataset shape and structure
- Checked for missing values

### 2. Data Preprocessing
- **Missing Value Handling**: Filled missing values with median for numeric columns
- **Feature Selection**: Separated features (X) and target variable (y = 'charges')
- **Categorical Encoding**: Applied OneHotEncoder to categorical variables:
  - `sex`
  - `smoker`
  - `region`
- **Train-Test Split**: Split data into 80% training and 20% testing sets
- **Feature Scaling**: Applied StandardScaler to normalize the features

### 3. Model Architecture

Built a sequential neural network with the following architecture:

```
Input Layer
    ↓
Dense Layer (12 neurons, ReLU activation)
    ↓
Dense Layer (24 neurons, ReLU activation)
    ↓
Dense Layer (48 neurons, ReLU activation)
    ↓
Dense Layer (96 neurons, ReLU activation)
    ↓
Output Layer (1 neuron, linear activation for regression)
```

### 4. Model Configuration
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Batch Size**: 32
- **Epochs**: 100

## 📈 Training Results

The model was trained for 100 epochs with the following performance progression:

### Initial Performance (Epoch 1)
- **Loss**: 322,392,896
- **MAE**: 13,343.80

### Final Performance (Epoch 100)
- **Loss**: ~24,000,000 (significant improvement)
- **MAE**: ~2,800 (substantial reduction in error)

### Training Progression
The model showed consistent improvement throughout training:
- **Early epochs**: Rapid decrease in both loss and MAE
- **Middle epochs**: Gradual refinement and fine-tuning
- **Final epochs**: Convergence with stable performance metrics

## 🔍 Key Observations

1. **Effective Learning**: The model successfully learned the patterns in the insurance data, with both loss and MAE decreasing consistently over epochs.

2. **Architecture Choice**: The progressive increase in neurons (12→24→48→96) allowed the model to capture increasingly complex patterns in the data.

3. **Data Preprocessing**: Proper handling of categorical variables through OneHotEncoding and feature scaling contributed to effective model training.

4. **Training Stability**: The Adam optimizer provided stable and efficient convergence throughout the training process.

## 📁 Files

- `insurance.ipynb`: Main Jupyter notebook containing the complete implementation
- `insurance.csv`: Dataset used for training and testing
- `README.md`: This documentation file

## 🚀 Requirements

```python
pandas
numpy
tensorflow
scikit-learn
```

## 💡 Usage

1. Open `insurance.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially to:
   - Load and preprocess the data
   - Build and compile the neural network
   - Train the model
   - Evaluate performance

## 🎯 Conclusion

This project demonstrates the application of deep learning for insurance cost prediction. The neural network architecture effectively captured the complex relationships between patient characteristics and insurance charges, achieving significant error reduction through training. The model can serve as a foundation for further optimization and deployment in real-world insurance pricing scenarios.
