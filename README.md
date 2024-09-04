# Facial Age Prediction using CNN and SVR Models

This project aims to predict the age of a person based on their facial images using a combination of Convolutional Neural Networks (CNNs) and Support Vector Regression (SVR). The project uses a dataset of facial images sorted into folders based on age. The CNN is employed for feature extraction from the images, and these features are then used by the SVR model to predict the age.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository

2. **Install the required Python packages:**
   Make sure you have Python installed, then install the necessary libraries using pip.
    
3. **Dataset Structure:**
   The dataset should be organized in a way where each folder is named after the age it represents, containing images of people of that age.

4. **Usage:**
   Run the script. Execute the Python script to load the data, train the models, and evaluate the results.
     ```bash
     python script.py

5. **Sample Output:** 
   After running the script, the Mean Absolute Error (MAE) of the predictions on the test set will be displayed. Additionally, a few sample predictions will be visualized with the actual and predicted ages.
    
## Results

The CNN model is used to extract features from facial images, which are then fed into an SVR model for age prediction. The model is evaluated using Mean Absolute Error (MAE).
The MAE score gives an indication of how accurate the predictions are compared to the actual ages. Lower MAE values indicate better performance.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.


