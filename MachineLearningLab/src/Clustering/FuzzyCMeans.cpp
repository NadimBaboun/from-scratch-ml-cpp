#include "FuzzyCMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <random> 
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  FuzzyCMeans class implementation  ///


// FuzzyCMeans function: Constructor for FuzzyCMeans class.//
FuzzyCMeans::FuzzyCMeans(int numClusters, int maxIterations, double fuzziness)
	: numClusters_(numClusters), maxIterations_(maxIterations), fuzziness_(fuzziness) {} //Original values in order: 3,100,2


// fit function: Performs Fuzzy C-Means clustering on the given dataset and return the centroids of the clusters.//
void FuzzyCMeans::fit(const std::vector<std::vector<double>>& data) {
	if (data.empty()) {
		throw std::runtime_error("Error: Empty dataset.");
	}
	if (numClusters_ <= 0) {
		throw std::invalid_argument("Error: Number of clusters must be positive.");
	}
	if (static_cast<size_t>(numClusters_) > data.size()) {
		throw std::invalid_argument("Error: Number of clusters cannot exceed number of data points.");
	}

	// Create a copy of the data to preserve the original dataset
	std::vector<std::vector<double>> normalizedData = data;

	/* Implement the following:
		--- Initialize centroids randomly
		--- Initialize the membership matrix with the number of data points
		--- Perform Fuzzy C-means clustering
	*/
	
	//TODO
	//Initialize centroids randomly
	std::mt19937 generator(std::random_device{}());
	std::uniform_int_distribution<int> distribution(0, static_cast<int>(data.size()) - 1);

	//resize it to match the "data" size
	centroids_.clear();
	centroids_.resize(numClusters_, std::vector<double>(data[0].size(), 0.0));

	std::vector<int> chosenIndices; //Store the chosen indices

	for (int j = 0; j < numClusters_; j++) {
		int randomIndex;

		do {
			randomIndex = distribution(generator);
		} while (std::find(chosenIndices.begin(), chosenIndices.end(), randomIndex) != chosenIndices.end());

		chosenIndices.push_back(randomIndex);
		centroids_[j] = data[randomIndex];
	}

	// Initialize the membership matrix with the number of data points
	initializeMembershipMatrix(static_cast<int>(data.size()));

	for (int iteration = 0; iteration < maxIterations_; iteration++) {
		//Update the membership matrix
		updateMembershipMatrix(data, centroids_);

		//Update the centroids
		std::vector<std::vector<double>> newCentroids = updateCentroids(data);

		//Update centroids with new values
		centroids_ = newCentroids;
	}
}



// initializeMembershipMatrix function: Initializes the membership matrix with random values that sum up to 1 for each data point.//
void FuzzyCMeans::initializeMembershipMatrix(int numDataPoints) {
	membershipMatrix_.clear();
	membershipMatrix_.resize(numDataPoints, std::vector<double>(numClusters_, 0.0));
	std::mt19937 generator(std::random_device{}());
	std::uniform_real_distribution<double> distribution(0.0, 1.0);

	/* Implement the following:
		--- Initialize membership matrix with random values that sum up to 1 for each data point
		---	Normalize membership values to sum up to 1 for each data point
	*/
	
	// TODO
	for (int i = 0; i < numDataPoints; i++) {
		double sum = 0.0;
		for (int j = 0; j < numClusters_; j++) {
			double randomValue = distribution(generator);
			membershipMatrix_[i][j] = randomValue;
			sum += membershipMatrix_[i][j];
		}
		//Normalize membership values
		if (sum > 0.0) {
			for (int j = 0; j < numClusters_; j++) {
				membershipMatrix_[i][j] /= sum;
			}
		}
		else {
			const double uniformMembership = 1.0 / numClusters_;
			for (int j = 0; j < numClusters_; j++) {
				membershipMatrix_[i][j] = uniformMembership;
			}
		}
	}
}


// updateMembershipMatrix function: Updates the membership matrix using the fuzzy c-means algorithm.//
void FuzzyCMeans::updateMembershipMatrix(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& centroids_) {

	/* Implement the following:
		---	Iterate through each data point
		--- Calculate the distance between the data point and the centroid
		--- Update the membership matrix with the new value
		--- Normalize membership values to sum up to 1 for each data point
	*/
	
	// TODO
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < numClusters_; j++) {
			double distance = SimilarityFunctions::euclideanDistance(data[i], centroids_[j]);
			double membershipValue;
			// Calculate the new membership value
			if (distance == 0.0) {
				membershipValue = 1.0;  //To avoid division by zero
			}
			else {
				membershipValue = 1.0 / (1.0 + pow(distance, 2.0 / (fuzziness_ - 1)));
			}
			membershipMatrix_[i][j] = membershipValue;
		}
	}

	//Normalize membership values to sum up to 1 for each data point
	for (int i = 0; i < data.size(); i++) {
		double sum = 0.0;
		for (int j = 0; j < numClusters_; j++) {
			sum += membershipMatrix_[i][j];
		}
		if (sum > 0.0) {
			for (int j = 0; j < numClusters_; j++) {
				membershipMatrix_[i][j] /= sum;
			}
		}
		else {
			const double uniformMembership = 1.0 / numClusters_;
			for (int j = 0; j < numClusters_; j++) {
				membershipMatrix_[i][j] = uniformMembership;
			}
		}
	}
}


// updateCentroids function: Updates the centroids of the Fuzzy C-Means algorithm.//
std::vector<std::vector<double>> FuzzyCMeans::updateCentroids(const std::vector<std::vector<double>>& data) {

	/* Implement the following:
		--- Iterate through each cluster
		--- Iterate through each data point
		--- Calculate the membership of the data point to the cluster raised to the fuzziness
	*/
	
	// TODO
	for (int j = 0; j < numClusters_; j++) {
		for (int k = 0; k < data[0].size(); k++) {
			double xSum = 0.0;
			double ySum = 0.0;

			for (int i = 0; i < data.size(); i++) {
				xSum += pow(membershipMatrix_[i][j], fuzziness_) * data[i][k];
				ySum += pow(membershipMatrix_[i][j], fuzziness_);
			}
			if (ySum > 0.0) {
				centroids_[j][k] = xSum / ySum;
			}

		}
	}

	return centroids_; //Return the centroids
}


// predict function: Predicts the cluster labels for the given data points using the Fuzzy C-Means algorithm.//
std::vector<int> FuzzyCMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels; // Create a vector to store the labels
	labels.reserve(data.size()); // Reserve space for the labels

	/* Implement the following:
		--- Iterate through each point in the data
		--- Iterate through each centroid
		--- Calculate the distance between the point and the centroid
		--- Calculate the membership of the point to the centroid
		--- Add the label of the closest centroid to the labels vector
	*/
	
	//TODO
	for (int i = 0; i < data.size(); i++) {
		int bestCluster = -1;
		double bestMembership = 0.0;
		for (int j = 0; j < numClusters_; j++) {
			if (membershipMatrix_[i][j] > bestMembership) {
				bestCluster = j;
				bestMembership = membershipMatrix_[i][j];
			}
		}
		labels.push_back(bestCluster);
	}

	return labels; //Return the labels vector

}


/// runFuzzyCMeans: this function runs the Fuzzy C-Means clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
FuzzyCMeans::runFuzzyCMeans(const std::string& filePath) {
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

		// Use the all dataset for training and testing sets.
		double trainRatio = 1.0;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData);

		// Make predictions on the training data
		std::vector<int> labels = predict(trainData);

		// Calculate evaluation metrics
		// Calculate Davies BouldinIndex using the actual features and predicted cluster labels
		double daviesBouldinIndex = Metrics::calculateDaviesBouldinIndex(trainData, labels);

		// Calculate Silhouette Score using the actual features and predicted cluster labels
		double silhouetteScore = Metrics::calculateSilhouetteScore(trainData, labels);


		// Create an instance of the PCADimensionalityReduction class
		PCADimensionalityReduction pca;

		// Perform PCA and project the data onto a lower-dimensional space
		int num_dimensions = 2; // Number of dimensions to project onto
		std::vector<std::vector<double>> reduced_data = pca.performPCA(trainData, num_dimensions);

		MessageBox::Show("Run completed");
		return std::make_tuple(daviesBouldinIndex, silhouetteScore, std::move(labels), std::move(reduced_data));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<int>(), std::vector<std::vector<double>>());
	}
}