#include "Metrics.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include "../Utils/SimilarityFunctions.h"


												///  Metrics class implementation  ///
			

// 1. Regression evaluation metrics //

// meanAbsoluteError function: Calculates the mean absolute error between two vectors of values.//
double Metrics::meanAbsoluteError(const std::vector<double>& trueValues, const std::vector<double>& predictedValues) {
	if (trueValues.size() != predictedValues.size()) {
		throw std::runtime_error("Error: trueValues and predictedValues have different sizes.");
	}
	if (trueValues.empty()) {
		throw std::runtime_error("Error: trueValues vector is empty.");
	}
	double mae = 0.0;
	size_t n = trueValues.size();
	for (size_t i = 0; i < n; i++) {
		mae += std::abs(trueValues[i] - predictedValues[i]);
	}
	return mae / static_cast<double>(n);
}


// rootMeanSquaredError function: Calculates the root mean squared error between two vectors of values.//
double Metrics::rootMeanSquaredError(const std::vector<double>& trueValues, const std::vector<double>& predictedValues) {
	if (trueValues.size() != predictedValues.size()) {
		throw std::runtime_error("Error: trueValues and predictedValues have different sizes.");
	}
	if (trueValues.empty()) {
		throw std::runtime_error("Error: trueValues vector is empty.");
	}
	double rmse = 0.0;
	size_t n = trueValues.size();
	for (size_t i = 0; i < n; i++) {
		double diff = trueValues[i] - predictedValues[i];
		rmse += diff * diff;
	}
	return std::sqrt(rmse / static_cast<double>(n));
}


/// rSquared function: Calculates the R-squared value of a given set of true values and predicted values.//
double Metrics::rSquared(const std::vector<double>& trueValues, const std::vector<double>& predictedValues) {
	if (trueValues.size() != predictedValues.size()) {
		throw std::runtime_error("Error: trueValues and predictedValues have different sizes.");
	}
	if (trueValues.empty()) {
		throw std::runtime_error("Error: trueValues vector is empty.");
	}
	double n = static_cast<double>(trueValues.size());
	double y_mean = 0.0;
	double SST = 0.0;
	double SSE = 0.0;

	// Calculate y_mean
	for (double y : trueValues) {
		y_mean += y;
	}
	y_mean /= n;

	// Calculate SST
	for (double y : trueValues) {
		SST += (y - y_mean) * (y - y_mean);
	}

	// Calculate SSE
	for (size_t i = 0; i < trueValues.size(); i++) {
		SSE += (trueValues[i] - predictedValues[i]) * (trueValues[i] - predictedValues[i]);
	}

	// Check for division by zero
	if (SST == 0.0) {
		throw std::runtime_error("Error: SST is zero.");
	}

	// Calculate R-squared
	double r_squared = 1.0 - (SSE / SST);
	return r_squared;
}



	// 2. Classification evaluation metrics //

	// accuracy function: Calculates the accuracy of the predicted labels compared to the true labels.//
	double Metrics::accuracy(const std::vector<double>&trueLabels, const std::vector<double>&predictedLabels) {
	if (trueLabels.size() != predictedLabels.size()) {
		throw std::runtime_error("Error: trueLabels and predictedLabels have different sizes.");
	}
	if (trueLabels.size() == 0) {
		throw std::runtime_error("trueLabels vector is empty, division by zero.");
	}

	int correct = 0;
	for (size_t i = 0; i < trueLabels.size(); ++i) {
		if (trueLabels[i] == predictedLabels[i]) {
			correct++;
		}
	}
	return static_cast<double>(correct) / static_cast<double>(trueLabels.size()) * 100.0;
}


// precision function: Calculates the precision of a given class label from the true labels and predicted labels.//
double Metrics::precision(const std::vector<double>& trueLabels, const std::vector<double>& predictedLabels, int classLabel) {
	if (trueLabels.size() != predictedLabels.size()) {
		throw std::runtime_error("Error: trueLabels and predictedLabels have different sizes.");
	}
	if (trueLabels.empty()) {
		throw std::runtime_error("Error: trueLabels vector is empty.");
	}
	int truePositive = 0, falsePositive = 0;
	for (size_t i = 0; i < trueLabels.size(); ++i) {
		if (predictedLabels[i] == classLabel) {
			if (trueLabels[i] == classLabel) {
				truePositive++;
			}
			else {
				falsePositive++;
			}
		}
	}
	if (truePositive + falsePositive == 0) {
		return 0;
	}
	else {
		return static_cast<double>(truePositive) / (truePositive + falsePositive) * 100;
	}
}


// recall function: Calculates the recall of a given class label from the true labels and predicted labels.//
double Metrics::recall(const std::vector<double>& trueLabels, const std::vector<double>& predictedLabels, int classLabel) {
	if (trueLabels.size() != predictedLabels.size()) {
		throw std::runtime_error("Error: trueLabels and predictedLabels have different sizes.");
	}
	if (trueLabels.empty()) {
		throw std::runtime_error("Error: trueLabels vector is empty.");
	}
	int truePositive = 0, falseNegative = 0;
	for (size_t i = 0; i < trueLabels.size(); ++i) {
		if (trueLabels[i] == classLabel) {
			if (predictedLabels[i] == classLabel) {
				truePositive++;
			}
			else {
				falseNegative++;
			}
		}
	}
	if (truePositive + falseNegative == 0) {
		return 0;
	}
	else {
		return static_cast<double>(truePositive) / (truePositive + falseNegative) * 100;
	}
}


// f1Score function: Calculates the F1 score for a given class label.//
double Metrics::f1Score(const std::vector<double>& trueLabels, const std::vector<double>& predictedLabels, int classLabel) {
	if (trueLabels.size() != predictedLabels.size()) {
		throw std::runtime_error("Error: trueLabels and predictedLabels have different sizes.");
	}
	if (trueLabels.empty()) {
		throw std::runtime_error("Error: trueLabels vector is empty.");
	}
	double precisionVal = precision(trueLabels, predictedLabels, classLabel);
	double recallVal = recall(trueLabels, predictedLabels, classLabel);
	if (precisionVal + recallVal == 0) {
		return 0;
	}
	else {
		return 2 * (precisionVal * recallVal) / (precisionVal + recallVal);
	}
}


/// confusionMatrix function: Calculates the confusion matrix given true labels and predicted labels.
std::vector<std::vector<int>> Metrics::confusionMatrix(const std::vector<double>& trueLabels, const std::vector<double>& predictedLabels, const int numClasses) {
	if (trueLabels.size() != predictedLabels.size()) {
		throw std::runtime_error("Error: trueLabels and predictedLabels have different sizes.");
	}
	if (numClasses <= 0) {
		throw std::runtime_error("Error: numClasses must be positive.");
	}
	std::vector<std::vector<int>> cm(numClasses, std::vector<int>(numClasses, 0));
	size_t n = trueLabels.size();

	for (size_t i = 0; i < n; i++) {
		int trueLabel = static_cast<int>(trueLabels[i]) - 1;
		int predictedLabel = static_cast<int>(predictedLabels[i]) - 1;

		// Perform error checking to ensure valid indices
		if (trueLabel >= 0 && trueLabel < numClasses && predictedLabel >= 0 && predictedLabel < numClasses) {
			cm[trueLabel][predictedLabel]++;
		}
		else {
			// Handle error case, e.g., throw an exception or print an error message
			throw std::runtime_error("Invalid labels encountered in trueLabels or predictedLabels vectors.");
		}
	}

	return cm;
}


// 3. Clustering evaluation metrics //

// calculateDaviesBouldinIndex function: Calculates the Davies-Bouldin index for a given set of data points and cluster labels.//
double Metrics::calculateDaviesBouldinIndex(const std::vector<std::vector<double>>& X, const std::vector<int>& labels) {
	if (X.empty() || X[0].empty()) {
		throw std::runtime_error("Error: Empty dataset.");
	}
	if (labels.empty() || labels.size() != X.size()) {
		throw std::runtime_error("Error: labels vector size does not match dataset size.");
	}
	std::vector<int> uniqueLabels;
	uniqueLabels.reserve(labels.size());
	for (int label : labels) {
		if (label < 0) {
			throw std::runtime_error("Error: Invalid cluster label.");
		}
		if (std::find(uniqueLabels.begin(), uniqueLabels.end(), label) == uniqueLabels.end()) {
			uniqueLabels.push_back(label);
		}
	}

	int numClusters = static_cast<int>(uniqueLabels.size());
	if (numClusters <= 1) {
		return 0.0;
	}
	std::vector<double> clusterSizes(numClusters, 0.0);
	std::vector<std::vector<double>> centroids(numClusters, std::vector<double>(X[0].size(), 0.0));

	// Calculate cluster sizes and centroids
	for (size_t i = 0; i < X.size(); ++i) {
		auto it = std::find(uniqueLabels.begin(), uniqueLabels.end(), labels[i]);
		if (it == uniqueLabels.end()) {
			throw std::runtime_error("Error: Invalid cluster label.");
		}
		int cluster = static_cast<int>(std::distance(uniqueLabels.begin(), it));
		const std::vector<double>& point = X[i];

		for (size_t j = 0; j < point.size(); ++j) {
			centroids[cluster][j] += point[j];
		}

		clusterSizes[cluster] += 1.0;
	}

	// Calculate cluster centroids by dividing by cluster sizes
	for (int i = 0; i < numClusters; ++i) {
		if (clusterSizes[i] > 0.0) {
			for (size_t j = 0; j < centroids[i].size(); ++j) {
				centroids[i][j] /= clusterSizes[i];
			}
		}
	}

	std::vector<double> clusterDiameters(numClusters, 0.0);
	for (int i = 0; i < numClusters; ++i) {
		const std::vector<double>& centroid = centroids[i];
		const int originalLabel = uniqueLabels[i];

		double maxDistance = 0.0;
		for (size_t j = 0; j < X.size(); ++j) {
			if (labels[j] == originalLabel) {
				double distance = SimilarityFunctions::euclideanDistance(X[j], centroid);
				if (distance > maxDistance) {
					maxDistance = distance;
				}
			}
		}

		clusterDiameters[i] = maxDistance;
	}

	double daviesBouldinIndex = 0.0;
	for (int i = 0; i < numClusters; ++i) {
		double maxIndex = 0.0;

		for (int j = 0; j < numClusters; ++j) {
			if (i != j) {
				double centroidDistance = SimilarityFunctions::euclideanDistance(centroids[i], centroids[j]);
				if (centroidDistance == 0.0) {
					continue;
				}
				double index = (clusterDiameters[i] + clusterDiameters[j]) / centroidDistance;
				if (index > maxIndex) {
					maxIndex = index;
				}
			}
		}

		daviesBouldinIndex += maxIndex;
	}

	daviesBouldinIndex /= numClusters;
	return daviesBouldinIndex;
}


// calculateSilhouetteScore function: Calculates the Silhouette Score for a given set of data points and labels.//
double Metrics::calculateSilhouetteScore(const std::vector<std::vector<double>>& X, const std::vector<int>& labels) {
	if (X.empty() || X[0].empty()) {
		throw std::runtime_error("Error: Empty dataset.");
	}
	if (labels.empty() || labels.size() != X.size()) {
		throw std::runtime_error("Error: labels vector size does not match dataset size.");
	}
	const size_t n = X.size();
	std::vector<int> uniqueLabels;
	uniqueLabels.reserve(labels.size());
	for (int label : labels) {
		if (label < 0) {
			throw std::runtime_error("Error: Invalid cluster label.");
		}
		if (std::find(uniqueLabels.begin(), uniqueLabels.end(), label) == uniqueLabels.end()) {
			uniqueLabels.push_back(label);
		}
	}

	if (uniqueLabels.size() <= 1) {
		return 0.0;
	}

	std::vector<std::vector<size_t>> clusterMembers(uniqueLabels.size());
	for (size_t i = 0; i < labels.size(); ++i) {
		auto it = std::find(uniqueLabels.begin(), uniqueLabels.end(), labels[i]);
		if (it == uniqueLabels.end()) {
			throw std::runtime_error("Error: Invalid cluster label.");
		}
		const size_t clusterIndex = static_cast<size_t>(std::distance(uniqueLabels.begin(), it));
		clusterMembers[clusterIndex].push_back(i);
	}

	double silhouetteScore = 0.0;
	for (size_t i = 0; i < n; ++i) {
		auto it = std::find(uniqueLabels.begin(), uniqueLabels.end(), labels[i]);
		if (it == uniqueLabels.end()) {
			throw std::runtime_error("Error: Invalid cluster label.");
		}
		const size_t ownCluster = static_cast<size_t>(std::distance(uniqueLabels.begin(), it));

		double a = 0.0;
		const auto& ownMembers = clusterMembers[ownCluster];
		if (ownMembers.size() > 1) {
			for (size_t idx : ownMembers) {
				if (idx == i) {
					continue;
				}
				a += SimilarityFunctions::euclideanDistance(X[i], X[idx]);
			}
			a /= static_cast<double>(ownMembers.size() - 1);
		}

		double b = std::numeric_limits<double>::max();
		for (size_t clusterIdx = 0; clusterIdx < clusterMembers.size(); ++clusterIdx) {
			if (clusterIdx == ownCluster || clusterMembers[clusterIdx].empty()) {
				continue;
			}

			double avgDistance = 0.0;
			for (size_t idx : clusterMembers[clusterIdx]) {
				avgDistance += SimilarityFunctions::euclideanDistance(X[i], X[idx]);
			}
			avgDistance /= static_cast<double>(clusterMembers[clusterIdx].size());

			if (avgDistance < b) {
				b = avgDistance;
			}
		}

		double s = 0.0;
		if (b != std::numeric_limits<double>::max()) {
			double denominator = std::max(a, b);
			if (denominator > 0.0) {
				s = (b - a) / denominator;
			}
		}

		silhouetteScore += s;
	}

	return silhouetteScore / static_cast<double>(n);
}



