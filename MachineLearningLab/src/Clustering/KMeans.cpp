#include "KMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  KMeans class implementation  ///

// KMeans function: Constructor for KMeans class.//
KMeans::KMeans(int numClusters, int maxIterations)
	: numClusters_(numClusters), maxIterations_(maxIterations) {}


// fit function: Performs K-means clustering on the given dataset and return the centroids of the clusters.//
void KMeans::fit(const std::vector<std::vector<double>>& data) {
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
		---	Initialize centroids randomly
		--- Randomly select unique centroid indices
		---	Perform K-means clustering
		--- Assign data points to the nearest centroid
		--- Calculate the Euclidean distance between the point and the current centroid
		--- Update newCentroids and clusterCounts
		--- Update centroids
		---  Check for convergence
	*/
	//store the centroid indexes in this
	std::vector<int> centroid_index;
	centroids_.clear();

	//loop for selecting the required centroids
	while (centroid_index.size() < static_cast<size_t>(numClusters_)) {
		//randomly select the indexes from the data
		int rand_index = std::rand() % data.size();
		//if the index has not been choosen before we basically add it to the centroid index list
		if (std::find(centroid_index.begin(), centroid_index.end(), rand_index) == centroid_index.end()) {
			centroid_index.push_back(rand_index);
		}
	}
	//this loop is for init the data points that correspond to the randomly choosen indexes
	for (int i = 0; i < numClusters_; i++) {
		centroids_.push_back(data[centroid_index[i]]);
	}
	//this is used for tracking convergence
	bool convergence = false;
	int count = 0;
	//loop until it converges or that the maximum amount of iterations have been achieved.
	while (!convergence && count < maxIterations_) {
		//init new centroids and keep track of how many data points are assigned to each cluster
		std::vector<std::vector<double>> newCentroids(numClusters_, std::vector<double>(data[0].size(), 0.0));
		std::vector<int> cluster_count(numClusters_, 0);
		
		//loop through each data point and give it to the nearest centroid
		for (const std::vector<double>& point : data) {
			double min_distance = (std::numeric_limits<double>::max)();
			int closest_centroid = -1;

			//find the closest centroid  for the current data point
			for (int i = 0; i < numClusters_; i++) {
				double distance = SimilarityFunctions::euclideanDistance(point, centroids_[i]);
				if (distance < min_distance) {
					min_distance = distance;
					closest_centroid = i;
				}
			}
			//update the new centroid position by adding current points coordinates
			for (int j = 0; j < point.size(); j++) {
				newCentroids[closest_centroid][j] += point[j];
			}

			//increase count of the cluster which the poit was assigned
			cluster_count[closest_centroid]++;
		}

		//calculate the new centorid positions by averaging the sum of all data points in the cluster
		for (int i = 0; i < numClusters_; i++) {
			for (int j = 0; j < centroids_[i].size(); j++) {
				if (cluster_count[i] != 0) {
					newCentroids[i][j] = newCentroids[i][j] / cluster_count[i];
				}
				else {
					newCentroids[i][j] = centroids_[i][j];
				}
			}
		}

		//check if the new centroid positions is matching the old ones, check for convergence
		convergence = true;
		for (int i = 0; i < numClusters_; i++) {
			for (int j = 0; j < centroids_[i].size(); j++) {
				if (std::abs(newCentroids[i][j] - centroids_[i][j]) > 1e-9) {
					convergence = false;
					break;
				}
			}
			if (!convergence) {
				break;
			}
		}
		//replace old centroids with new centroids and then incrase count
		centroids_ = newCentroids;
		count++;
		
	}
	// TODO
}


//// predict function: Calculates the closest centroid for each point in the given data set and returns the labels of the closest centroids.//
std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels;
	labels.reserve(data.size());

	if (centroids_.empty()) {
		throw std::logic_error("Error: Model has not been fitted.");
	}
	
	/* Implement the following:
		--- Initialize the closest centroid and minimum distance to the maximum possible value
		--- Iterate through each centroid
		--- Calculate the Euclidean distance between the point and the centroid
		--- Add the closest centroid to the labels vector
    */
	// TODO

	//loop thorygugh each of the data points in the dataset
	for (const std::vector<double>& point : data) {
		//init the closest centroid to an invalid index value and init the min distance using max value of double
		int closest_centroid = -1;
		double min_distance = (std::numeric_limits<double>::max)();
		//go thoruhg all centroids to find the closest to the current data point
		for (int i = 0; i < numClusters_; ++i) {
			double distance = SimilarityFunctions::euclideanDistance(point, centroids_[i]);
			//check if the euclidean distance calculated is less then the current min distance then update the min distance
			if (distance < min_distance) {
				min_distance = distance;
				closest_centroid = i;
			}
		}
		//basically to put value of closest centroid into current data point
		labels.push_back(closest_centroid);
	}
	return labels; // Return the labels vector

}





/// runKMeans: this function runs the KMeans clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
KMeans::runKMeans(const std::string& filePath) {
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
