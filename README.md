# A Hybrid Model for Breast Cancer Classification

This repository contains the code and resources for the project **"A Hybrid Model for Breast Cancer Classification,"** submitted in partial fulfillment of the requirements for the Master of Science in Computer Science at the University of Mysore.

## Project Description

Breast cancer is a critical global health issue, responsible for millions of deaths worldwide. Early detection and accurate classification of breast cancer into normal, benign, and malignant categories are vital for effective treatment. This project enhances breast cancer detection using a hybrid approach that combines traditional machine learning (ML) classifiers and deep learning (DL) models.

The study compares the performance of several ML classifiers and DL models and builds hybrid models by integrating the best-performing classifiers from each approach.

## Objectives

- To analyze the efficiency of machine learning algorithms in breast cancer classification.
- To compare deep learning architectures for breast cancer identification.
- To propose a hybrid model that combines the strengths of both ML and DL methods for optimal performance.

## Datasets Used

- **Mini-MIAS Database**: Contains 322 mammogram images classified as normal, benign, or malignant.
- **Mini-DDSM Database**: Contains a large number of mammographic images (53 GB) from two views, with detailed notes on abnormalities.

The combined dataset initially contained 8,276 images. After applying data augmentation techniques, the dataset was expanded to 9,000 images, with 3,000 images belonging to each class (normal, benign, and malignant).

## Methodology

### Machine Learning Classifiers

Various machine learning classifiers were trained to identify the best-performing model for breast cancer classification, including:

- Logistic Regression
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)
- Decision Trees
- Random Forest
- XGBoost

The highest accuracy achieved by machine learning classifiers was **84%**.

### Deep Learning Models

Several pretrained Keras models were trained to compare their performance, including:

- DenseNet201
- VGG16
- ResNet50

The highest accuracy achieved by deep learning models was **86%**.

### Hybrid Model

After identifying the best-performing classifiers from both the machine learning and deep learning models, hybrid models were built by combining them. The best-performing hybrid model is DenseNet201 for feature extraction, combined with an ensemble of SVM and XGBoost for classification. The hybrid model achieved an overall accuracy of **90%**.

## Results

The hybrid model, which combines DenseNet201 with an ensemble of SVM and XGBoost, outperforms the individual models by achieving an overall accuracy of **90%**. The class-wise accuracy of the hybrid model is:

- Normal: 87%
- Benign: 84%
- Malignant: 98%

## Challenges

- **Normal vs. Benign Classification**: Normal and benign mammogram images have similar tissue patterns, making them difficult to differentiate.
- **Imbalanced Data**: The original dataset had more images for malignant cases compared to normal and benign cases, which was addressed using data augmentation techniques.

## Tools and Technologies Used

- Python 3.x
- TensorFlow
- Keras (for pretrained models)
- Scikit-learn
- XGBoost
- Seaborn
- Matplotlib

## Conclusion

Our study expands breast cancer classification by including a third class, normal, alongside benign and malignant. After evaluating both machine learning and deep learning models, the hybrid model combining DenseNet201 with an ensemble of SVM and XGBoost achieved the best performance, with an overall accuracy of 90%. This demonstrates the effectiveness of the hybrid approach in improving breast cancer classification performance.

## Future Work

- Extend the model to work with larger and more diverse datasets.
- Improve classification accuracy through additional data preprocessing and augmentation techniques.

## Author

**Martin Manoj Kumar S**

## License

This project is currently not licensed.
