# Powerlifting Data Analysis Project

This project analyzes powerlifting competition data to understand various aspects of the sport, including performance patterns, feature importance, and predictive modeling.

## Project Structure

The project consists of several Python scripts and Jupyter notebooks that perform different analyses:

1. **Data Selection and Sampling** (`data_selection.py`, `data_selection.ipynb`)
   - Handles initial data loading and sampling
   - Creates a representative subset of the dataset

2. **Data Preprocessing** (`preprocessing_summary.py`, `preprocessing_summary.ipynb`)
   - Performs data cleaning and preprocessing
   - Generates summary statistics and distribution plots

3. **Exploratory Data Analysis** (`eda_visuals.py`, `eda_visuals.ipynb`)
   - Creates correlation heatmaps
   - Generates boxplots for weight analysis
   - Visualizes federation and country distributions

4. **Hierarchical Clustering** (`clustering_hierarchical.py`, `clustering_hierarchical.ipynb`)
   - Performs hierarchical clustering analysis
   - Creates dendrograms for both raw and standardized data

5. **Regression Analysis** (`regression.py`, `regression.ipynb`)
   - Implements various regression models
   - Evaluates model performance using cross-validation

6. **Principal Component Analysis** (`pca_analysis.py`, `pca_analysis.ipynb`)
   - Performs PCA on the dataset
   - Visualizes feature contributions
   - Analyzes explained variance

7. **Feature Selection** (`feature_selection_analysis.py`, `feature_selection_analysis.ipynb`)
   - Analyzes feature importance using multiple methods
   - Compares different feature selection approaches

8. **Classification Analysis** (`classification.py`, `classification.ipynb`)
   - Implements various classification models
   - Evaluates model performance
   - Analyzes class balance and overfitting

## Setup Instructions

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data

The project uses the OpenPowerlifting dataset. The main dataset file is `openpowerlifting.csv`, and a sampled version is created as `sample_openpowerlifting.csv` for analysis.

## Usage

Each analysis can be run either through the Python scripts or Jupyter notebooks:

1. To run Python scripts:
   ```bash
   python [script_name].py
   ```

## Output Files

The analysis generates several output files:
- PNG files for various visualizations
- Summary reports in text format
- Model evaluation results

## Dependencies

The project requires the following Python packages (see `requirements.txt` for versions):
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
- jupyter
- nbformat
