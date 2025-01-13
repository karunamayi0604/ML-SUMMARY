# **Machine Learning Model Evaluator**

## **Overview**
This project serves as a **comprehensive toolkit for machine learning**, providing an interactive interface to evaluate multiple machine learning models with minimal effort. Built using **Streamlit** and **scikit-learn**, it allows users to upload datasets, preprocess data, and compare the performance of popular classification algorithms—all in a single app.

## **Features**
1. **Upload Your Dataset**:
   - Accepts CSV files for easy integration with your data.
   - Automatically previews the dataset for verification.

2. **Customizable Input**:
   - Select target and feature columns interactively.
   - Split data into training and testing sets automatically.

3. **Wide Range of Models**:
   - Includes a variety of machine learning models:
     - **Linear Models**: Logistic Regression, Ridge Classifier, SGD Classifier.
     - **Support Vector Machines**: SVC.
     - **Tree-Based Models**: Decision Tree, Random Forest, Gradient Boosting, AdaBoost.
     - **Nearest Neighbors**: K-Nearest Neighbors (KNN).
     - **Bayesian Models**: GaussianNB.
     - **Discriminant Analysis**: Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA).
   - New models can be easily added.

4. **Performance Metrics**:
   - Evaluates models using:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
   - Displays results in a user-friendly table.

5. **Visualizations**:
   - Bar charts for model comparison.
   - Confusion matrix heatmaps for detailed performance insights.

## **How It Works**
1. Upload your dataset (CSV).
2. Select the target column and feature columns.
3. Choose the models you wish to evaluate.
4. The app:
   - Splits the data into training and testing sets.
   - Trains each model on the training data.
   - Evaluates models on the testing data.
   - Displays metrics and visualizations.

## **Tech Stack**
- **Streamlit**: For building the interactive web app.
- **scikit-learn**: For machine learning model implementation and evaluation.
- **Pandas**: For data manipulation.
- **Seaborn** & **Matplotlib**: For visualizing results.

## **Why This Project?**
This app simplifies machine learning experimentation, acting as a **gist of the entire ML process**:
- Data preprocessing
- Model training and evaluation
- Metrics calculation and visualization
It’s perfect for beginners to grasp ML concepts and for experts to benchmark datasets quickly.

## **How to Run**
1. Clone this repository:
   ```bash
   git clone <repo-url>
   ```
2. Install dependencies:
   ```bash
   pip install matlplotlib
   pip install skit-learn
   pip install numpy
   pip install pandas
   pip install streamlit
   ```
3. Run the app:
   ```bash
   streamlit run modeleval.py
   ```

## **Future Enhancements**
- Add hyperparameter tuning for models.
- Include regression models and metrics.
- Support for deep learning models (e.g., TensorFlow, PyTorch).

---

Would you like to add installation steps for specific environments or additional sections?
