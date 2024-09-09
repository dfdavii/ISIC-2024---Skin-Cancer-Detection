# ISIC 2024 - Skin Cancer Detection with 3D-TBP: using tabular data

## Data Preparation and Feature Engineering

### Align Columns
The `align_columns` function ensures that the training and test DataFrames have the same columns. It drops any extra columns in the test DataFrame that are not present in the training DataFrame and aligns the columns in the same order, excluding the 'target' column.

### Create Features
The `create_features` function generates new features for both the training and test DataFrames. It calculates various ratios and contrasts based on existing columns, such as `lesion_size_ratio`, `perimeter_area_ratio`, `color_contrast_AB`, `color_contrast_LB`, and `symmetry_asymmetry_ratio`. It also drops the original columns used to create these new features.

## Data Loading and Preprocessing

### Load Data
The `load_data` function reads the training and test data from CSV files and returns them as DataFrames.

### Select Columns
The `select_columns` function selects specific columns from the training and test DataFrames based on the provided lists of selected columns.

### Fill Missing Values
The `fill_missing_values` function fills missing values in the DataFrame columns with the mode (most frequent value) of each column.

### Encode and Scale
The `encode_and_scale` function performs one-hot encoding on categorical columns and standard scaling on numerical columns. It concatenates the encoded and scaled columns back to the original DataFrames and ensures that all float64 columns are converted to float32.

## Candidate Information

### Candidate Info Tuple
The `CandidateInfoTuple` is a named tuple that stores information about each candidate, including whether it is a nodule, the ISIC ID, the patient ID, and the attributes.

### Get Candidate Info List
The `getCandidateInfoList` function generates a list of `CandidateInfoTuple` objects from a DataFrame. It drops unnecessary columns, extracts attributes, and determines the value of `isNodule_bool` based on the 'target' column. The list is sorted in reverse order.

## Image Handling

### Img Class
The `Img` class handles image data from an HDF5 file. It reads the image data, applies optional transformations, and converts the image to a NumPy array. It also normalizes the image data if necessary.

## TensorFlow Dataset

### ImgDatasetTF Class
The `ImgDatasetTF` class creates a TensorFlow dataset for training and validation. It processes the DataFrame in chunks, filters by ISIC ID if provided, handles validation set creation, and sorts the data. It also supports shuffling and balancing the ratio of positive and negative samples.

### Transformations
The `transform_train` function applies random horizontal flips and normalization to training images. The `transform_test` function resizes and normalizes test images.

### Display Image
The `display_image_from_dataset` function loads an example image from the dataset and displays it using Matplotlib.

## Model Training and Validation

### Calculate Metrics
The `calculate_metrics` function calculates various metrics, including confusion matrix, precision, recall, F1 score, and accuracy, based on the true and predicted labels.

### Train Model
The `train_model` function trains the model using the provided training and validation datasets. It includes early stopping based on validation loss and calculates metrics for each epoch.

## Custom Model

### MyModel Class
The `MyModel` class defines a custom TensorFlow model for binary classification based on attributes. It includes methods for forward pass, configuration, and reconstruction from configuration.

## Test Dataset

### TestDataset Class
The `TestDataset` class handles test data and generates a TensorFlow dataset for predictions. It includes a generator function to yield ISIC ID, patient ID, and attributes.

### Predict Function
The `predict` function makes predictions using the model on the test dataset. It calculates probabilities, extracts malignant class probabilities, and rounds them to one decimal place. It also handles garbage collection to manage memory usage.



