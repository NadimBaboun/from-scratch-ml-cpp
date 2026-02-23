#include "LinearRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <unordered_map>
#include <msclr\marshal_cppstd.h>
#include <stdexcept>
#include "../MainForm.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace System::Windows::Forms; // For MessageBox



										///  LinearRegression class implementation  ///

//function overloading, uses gradient descent.
//Only Fit function needed to be overloaded.
//Overloaded with epoch and learning rate
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels, double learning_rate, int num_epochs) {

    if (trainData.size() != trainLabels.size()) {
        throw std::logic_error("Size of trainData and trainLabels do not match!");
    }

    //Variable decleration
    int numFeatures = trainData[0].size();
    int numSamples = trainData.size();

    //Initialize coefficients with zeros
    m_coefficients = Eigen::VectorXd::Zero(numFeatures + 1); //"+1" used to make space for bias term

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Loop over training examples
        for (int i = 0; i < numSamples; i++) {
            Eigen::VectorXd input(numFeatures + 1); //"+1" used to make space for bias term
            input[0] = 1.0; //Set the bias term

            for (int j = 1; j <= numFeatures; j++) {
                input[j] = trainData[i][j - 1]; //Subtract 1 because arrays start at 0
            }

            // Calculate the predicted value
            double predictedValue = 0.0;
            for (int j = 0; j < input.size(); j++) {
                predictedValue += input[j] * m_coefficients[j];
            }

            //Calculate the error
            double error = predictedValue - trainLabels[i];

            //Update coefficients using gradient descent
            for (int j = 0; j < input.size(); j++) {
                m_coefficients[j] -= learning_rate * error * input[j];
            }
        }
    }
}

/*std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData, double learning_rate) {
    int numSamples = testData.size();
    int numFeatures = testData[0].size();
    std::vector<double> predictions(numSamples);

    for (int i = 0; i < numSamples; i++) {
        Eigen::VectorXd input(numFeatures + 1);
        input[0] = 1.0; // Set the bias term

        for (int j = 1; j <= numFeatures; j++) {
            input[j] = testData[i][j - 1]; //Subtract 1 because arrays start at 0
        }

        //Calculate the predicted value (dot product)
        double predictedValue = 0.0;
        for (int j = 0; j < input.size(); j++) {
            predictedValue += input[j] * m_coefficients[j];
        }

        predictions[i] = predictedValue;
    }

    return predictions;
}*/

// Function to fit the linear regression model to the training data //
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) {

	// This implementation is using Matrix Form method
	/* Implement the following:	  
	    --- Check if the sizes of trainData and trainLabels match
	    --- Convert trainData to matrix representation
	    --- Construct the design matrix X
		--- Convert trainLabels to matrix representation
		--- Calculate the coefficients using the least squares method
		--- Store the coefficients for future predictions
	*/
    if (trainData.size() != trainLabels.size()) {
        throw std::logic_error("Size of trainData and trainLabels do not match!");
    }

    //Variable decleration
    int numSamples = trainData.size();
    int numFeatures = trainData[0].size();

    //Convert trainData to matrix representation
    Eigen::MatrixXd X(numSamples, numFeatures);
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            X(i, j) = trainData[i][j];
        }
    }

    //Construct the design matrix X
    Eigen::MatrixXd X_design(numSamples, numFeatures + 1);
    X_design << Eigen::MatrixXd::Ones(numSamples, 1), X; // Horizontal concatenation

    //Convert trainLabels to matrix representation
    Eigen::VectorXd y(numSamples);
    for (int i = 0; i < numSamples; i++) {
        y(i) = trainLabels[i];
    }

    //Calculate the Gram matrix (X^T * X)
    Eigen::MatrixXd gram_matrix = X_design.transpose() * X_design;

    //Perform LDLT decomposition on the Gram matrix
    Eigen::LDLT<Eigen::MatrixXd> ldlt_decomposition(gram_matrix);

    //Calculate the right-hand side of the linear system (X^T * y)
    Eigen::VectorXd rhs = X_design.transpose() * y;

    //Solve the linear system to obtain the coefficients (A.solve(B))
    Eigen::VectorXd coefficients = ldlt_decomposition.solve(rhs);

    //Store the coefficients for future predictions
    m_coefficients = coefficients;
}

// Function to make predictions on new data //
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {

	// This implementation is using Matrix Form method    
    /* Implement the following
		--- Check if the model has been fitted
		--- Convert testData to matrix representation
		--- Construct the design matrix X
		--- Make predictions using the stored coefficients
		--- Convert predictions to a vector
	*/
	
	// TODO
    if (m_coefficients.size() == 0) {
        throw std::logic_error("Model has not been fitted!");
    }
    //Variable decleration
    int numSamples = testData.size();
    int numFeatures = testData[0].size();

    //Convert testData to matrix 
    Eigen::MatrixXd X(numSamples, numFeatures);

    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            X(i, j) = testData[i][j];
        }
    }

    //Construct the design matrix X_design
    Eigen::MatrixXd X_design(numSamples, numFeatures + 1);
    X_design << Eigen::MatrixXd::Ones(numSamples, 1), X; // Horizontal concatenation

    //Make predictions using the stored coefficients, vector multiplication
    Eigen::VectorXd predictions = X_design * m_coefficients;

    //Convert predictions to a vector
    std::vector<double> result(numSamples);
    for (int i = 0; i < numSamples; i++) {
        result[i] = predictions(i);
    }

    return result;
}



/// runLinearRegression: this function runs the Linear Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets. ///

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    LinearRegression::runLinearRegression(const std::string& filePath, int trainingRatio) {
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
        // Load the dataset from the file path
        std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

        // Convert the dataset from strings to doubles
        std::vector<std::vector<double>> dataset;
        bool isFirstRow = true; // Flag to identify the first row

        for (const auto& row : data) {
            if (isFirstRow) {
                isFirstRow = false;
                continue; // Skip the first row (header)
            }

            std::vector<double> convertedRow;
            for (const auto& cell : row) {
                try {
                    double value = std::stod(cell);
                    convertedRow.push_back(value);
                }
                catch (const std::exception&) {
                    // Handle the exception or set a default value
                    std::cerr << "Error converting value: " << cell << std::endl;
                    // You can choose to set a default value or handle the error as needed
                }
            }
            dataset.push_back(convertedRow);
        }

        // Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
        double trainRatio = trainingRatio * 0.01;

        std::vector<std::vector<double>> trainData;
        std::vector<double> trainLabels;
        std::vector<std::vector<double>> testData;
        std::vector<double> testLabels;

        DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

        // Fit the model to the training data

        //fit(trainData, trainLabels);
        fit(trainData, trainLabels, 0.000003, 100); //Use this to utilize the overloading function

        // Make predictions on the test data

        std::vector<double> testPredictions = predict(testData);

        // Calculate evaluation metrics (e.g., MAE, MSE)
        double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
        double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
        double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

        // Make predictions on the training data

        std::vector<double> trainPredictions = predict(trainData);

        // Calculate evaluation metrics for training data
        double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
        double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
        double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);

        MessageBox::Show("Run completed");
        return std::make_tuple(test_mae, test_rmse, test_rsquared,
            train_mae, train_rmse, train_rsquared,
            std::move(trainLabels), std::move(trainPredictions),
            std::move(testLabels), std::move(testPredictions));
    }
    catch (const std::exception& e) {
        // Handle the exception
        MessageBox::Show("Not Working");
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            std::vector<double>(), std::vector<double>(),
            std::vector<double>(), std::vector<double>());
    }
}