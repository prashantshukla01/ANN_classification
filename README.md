# Customer Churn Prediction using ANN

This project is a Streamlit-based web application that predicts customer churn probability using a trained Artificial Neural Network (ANN) model. It allows users to input customer details and get a real-time prediction on whether the customer is likely to leave the bank.

## Features

-   **Interactive User Interface**: Built with Streamlit for easy data input.
-   **Real-time Prediction**: Instantly calculates churn probability based on user inputs.
-   **Data Preprocessing**: Handles categorical data (Gender, Geography) and scales numerical features using pre-trained encoders and scalers.
-   **Deep Learning Model**: Utilizes a TensorFlow/Keras ANN model for accurate predictions.

## Tech Stack

-   **Python**: Core programming language.
-   **Streamlit**: Web framework for the UI.
-   **TensorFlow / Keras**: Deep learning framework for the ANN model.
-   **Scikit-learn**: For data preprocessing (Label Encoding, One-Hot Encoding, Scaling).
-   **Pandas**: Data manipulation and analysis.
-   **NumPy**: Numerical operations.

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory:
    ```bash
    cd /path/to/ann_classification
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2.  **Interact with the App**:
    -   Open your web browser (usually at `http://localhost:8501`).
    -   Fill in the customer details such as Geography, Gender, Age, Balance, Credit Score, etc.
    -   The app will display the **Churn Probability** and a message indicating whether the customer is likely to churn.

## Project Structure

-   `app.py`: The main Streamlit application script.
-   `model.h5`: The pre-trained ANN model.
-   `label_encoder_gender.pkl`: Pickle file for the Gender label encoder.
-   `one_hot_encoder_geo.pkl`: Pickle file for the Geography one-hot encoder.
-   `scaler.pkl`: Pickle file for the StandardScaler.
-   `requirements.txt`: List of Python dependencies.
-   `experiments.ipynb`: Jupyter notebook used for training and experimenting with the model.
-   `prediction.ipynb`: Jupyter notebook for testing predictions.
-   `Churn_Modelling.csv`: Dataset used for training (if available).

## Model Details

The model is an Artificial Neural Network (ANN) trained on the Churn Modelling dataset. It takes various customer attributes as input, preprocesses them (encoding categorical variables and scaling numerical ones), and outputs a probability score between 0 and 1. A score > 0.5 indicates a high likelihood of churn.
