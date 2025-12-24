# 🎵 Music Recommendation System

This project builds a simple Machine Learning-based Music Recommendation System. It uses a **Decision Tree Classifier** to predict the genre of music a user is likely to enjoy based on their age and gender.

## 📂 Project Overview

The system analyzes a dataset of user profiles and their preferred music genres to learn patterns. It then uses these patterns to make recommendations for new users.

**Key Steps:**
1.  **Data Importing**: Loading the user data.
2.  **Data Preparation**: Splitting data into input features (Age, Gender) and output targets (Genre).
3.  **Model Training**: Using a Decision Tree algorithm to learn from the data.
4.  **Evaluation**: Measuring the accuracy of the model using a test set.
5.  **Persistence**: Saving the trained model using `joblib` so it can be reused without retraining.
6.  **Visualization**: Visualizing the decision tree logic using Graphviz.

## 📊 Dataset

The project uses a file named `music.csv` containing basic demographic information and preferences.

**Columns:**
* `age`: The age of the user (Integer).
* `gender`: The gender of the user (1 = Male, 0 = Female).
* `genre`: The music genre preferred by the user (Target variable, e.g., HipHop, Jazz, Classical, Dance, Acoustic).

## 🛠️ Technologies & Libraries Used

The project is implemented in Python using a Jupyter Notebook. Key libraries include:

* **Pandas**: For data manipulation and loading CSV files.
* **Scikit-learn**:
    * `DecisionTreeClassifier`: The core algorithm used for classification.
    * `train_test_split`: To split the data into training and testing sets.
    * `accuracy_score`: To evaluate model performance.
    * `tree`: To export the decision tree structure.
* **Joblib**: For saving (`dump`) and loading (`load`) the trained model file (`music-recommender.joblib`).
* **Graphviz**: To visualize the decision tree structure from the `.dot` file.

## 🧠 Model Logic

The model uses a **Decision Tree** to classify users.
* It splits the data based on **Age** thresholds (e.g., <= 25, <= 30) and **Gender**.
* For example, it might learn that users aged 20-25 are more likely to enjoy *HipHop* (if Male) or *Dance* (if Female), while older demographics might prefer *Classical* or *Jazz*.

## 🚀 How to Run

1.  Clone this repository.
2.  Ensure you have the required libraries installed:
    ```bash
    pip install pandas scikit-learn joblib graphviz
    ```
3.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "01_Music Recommendation System.ipynb"
    ```
4.  Run the cells to train the model and see predictions.

**Example Prediction:**
The model can predict the preferred genre for a **21-year-old Male**:
```python
model.predict([[21, 1]])
# Output: array(['HipHop'], dtype=object)
