#include "LogisticRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <string>
#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <unordered_map> 

using namespace System::Windows::Forms; // For MessageBox


                                            ///  LogisticRegression class implementation  ///
// Constractor

LogisticRegression::LogisticRegression(double learning_rate, int num_epochs)
    : learning_rate(learning_rate), num_epochs(num_epochs) {}
// Fit method for training the logistic regression model
void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    int num_features = static_cast<int>(X_train[0].size());
    int num_classes = static_cast<int>(std::set<double>(y_train.begin(), y_train.end()).size());

    //init weights with num_classes and num_features and assign value of 0.5
    weights = std::vector<std::vector<double>>(num_classes, std::vector<double>(num_features, 0.5));

    /* Implement the following:
        --- Initialize weights for each class
        --- Loop over each class label
        --- Convert the problem into a binary classification problem
        --- Loop over training epochs
        --- Add bias term to the training example
        --- Calculate weighted sum of features
        --- Calculate the sigmoid of the weighted sum
        --- Update weights using gradient descent
    */
    for (int class_label = 1; class_label <= num_classes; class_label++) {
        //converting into binary the classes 1,2,3 to 1,0
        std::vector<double> binaryLabels;
        for (double label : y_train) {
            //add 1 if label is equal to class_label
            if (label == class_label) {
                binaryLabels.push_back(1);
            }
            //Otherwise add 0
            else {
                binaryLabels.push_back(0);
            }
        }
        //perform training for a number of epochs
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            //loop through each training point/example in the dataset
            for (size_t i = 0; i < X_train.size(); i++) {
                //Extract the features for the current training example
                //add bias term to training example
                const std::vector<double>& features = X_train[i];

                // calculate the weighted sum of feature using the current weights
                double weighted_sum = 0.0;
                for (int j = 0; j < num_features; j++) {
                    weighted_sum += weights[class_label - 1][j] * features[j];
                }

                //sigmoid calculation using weighted sum
                double sigmoid_output = sigmoid(weighted_sum);

                // Update weights using gradient descent for each feature
                for (int j = 0; j < num_features; j++) {
                    //calculate the difference between the true label and sigmoid output
                    double diff = binaryLabels[i] - sigmoid_output;
                    weights[class_label - 1][j] += learning_rate * diff * sigmoid_output * (1.0 - sigmoid_output) * features[j];
                }
            }
        }
    }

}

// Predict method to predict class labels for test data
std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;
    //loop through all vectors in X_test
    for (const std::vector<double>& sample : X_test) {
        std::vector<double> class_probabilities;
        //then loop through all classes in weights vector
        for (int class_label = 1; class_label <= static_cast<int>(weights.size()); class_label++) {
            
            //calculte the weighted sum of features
            double weighted_sum = 0.0;
            for (size_t j = 0; j < sample.size(); j++) {
                weighted_sum += weights[class_label - 1][j] * sample[j];
            }

            
            //afterwards calculate the sigmoid of the weighted sum and store that value
            double sigmoid_output = sigmoid(weighted_sum);
            class_probabilities.push_back(sigmoid_output);
        }

        //find the class with the highest probability and assign it as the prediction, 
        int predicted_class = 1;
        double max_probability = class_probabilities[0];
        for (int class_label = 2; class_label <= static_cast<int>(weights.size()); class_label++) {
            //if the statement is not fulfilled that would mean that the predicted class is equal to 1
            //and tahts why the class_label is set too 2 instead of 1.
            if (class_probabilities[class_label - 1] > max_probability) {
                predicted_class = class_label;
                max_probability = class_probabilities[class_label - 1];
            }
        }

        //Store the predicted classes (1, 2, 3) in the predictions vector
        predictions.push_back(static_cast<double>(predicted_class));
    }

    return predictions;
}

/// runLogisticRegression: this function runs the logistic regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> 
LogisticRegression::runLogisticRegression(const std::string& filePath, int trainingRatio) {

    try {
        // Check if the file path is empty
        if (filePath.empty()) {
            MessageBox::Show("Please browse and select the dataset file from your PC.");
            return {}; // Return an empty vector since there's no valid file path
        }

        // Attempt to open the file
        std::ifstream file(filePath);
        if (!file.is_open()) {
            MessageBox::Show("Failed to open the dataset file");
            return {}; // Return an empty vector since file couldn't be opened
        }

        std::vector<std::vector<double>> dataset; // Create an empty dataset vector
        DataLoader::loadAndPreprocessDataset(filePath, dataset);

        // Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
        double trainRatio = trainingRatio * 0.01;

        std::vector<std::vector<double>> trainData;
        std::vector<double> trainLabels;
        std::vector<std::vector<double>> testData;
        std::vector<double> testLabels;

        DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

        // Fit the model to the training data
        fit(trainData, trainLabels);

        // Make predictions on the test data
        std::vector<double> testPredictions = predict(testData);

        // Calculate accuracy using the true labels and predicted labels for the test data
        double test_accuracy = Metrics::accuracy(testLabels, testPredictions);

        // Make predictions on the training data
        std::vector<double> trainPredictions = predict(trainData);

        // Calculate accuracy using the true labels and predicted labels for the training data
        double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

        MessageBox::Show("Run completed");
        return std::make_tuple(train_accuracy, test_accuracy,
            std::move(trainLabels), std::move(trainPredictions),
            std::move(testLabels), std::move(testPredictions));
    }
    catch (const std::exception& e) {
        // Handle the exception
        MessageBox::Show("Not Working");
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return std::make_tuple(0.0, 0.0, std::vector<double>(),
            std::vector<double>(), std::vector<double>(),
            std::vector<double>());
    }
}
