#include "DataPreprocessor.h"
#include "DataLoader.h"
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <Windows.h> 
#include <string>    
using namespace System::Windows::Forms; // For MessageBox




//// Function to split dataset into training and testing sets
void DataPreprocessor::splitDataset(const std::vector<std::vector<double>>& dataset, double trainRatio,
	std::vector<std::vector<double>>& trainData, std::vector<double>& trainLabels,
	std::vector<std::vector<double>>& testData, std::vector<double>& testLabels) {
	trainData.clear();
	trainLabels.clear();
	testData.clear();
	testLabels.clear();

	if (dataset.empty()) {
		throw std::runtime_error("Error: Empty dataset.");
	}

	if (trainRatio < 0.0 || trainRatio > 1.0) {
		throw std::invalid_argument("Error: Train ratio must be between 0 and 1.");
	}

	// Calculate the number of training samples based on the trainRatio
	size_t trainSize = static_cast<size_t>(dataset.size() * trainRatio);

	// Extract the features and labels from the dataset
	std::vector<std::vector<double>> features;
	std::vector<double> labels;
	for (const auto& row : dataset) {
		if (row.empty()) {
			continue;  // Skip empty rows
		}

		if (row.size() < 2) {
			throw std::runtime_error("Error: Row does not have enough elements for features and label.");
		}

		std::vector<double> featureRow(row.begin(), row.end() - 1);
		double label = row.back();
		features.push_back(featureRow);
		labels.push_back(label);
	}

	if (features.empty()) {
		throw std::runtime_error("Error: No valid rows found in dataset.");
	}

	const size_t expectedFeatureCount = features[0].size();
	for (size_t i = 1; i < features.size(); ++i) {
		if (features[i].size() != expectedFeatureCount) {
			throw std::runtime_error("Error: Inconsistent number of features across rows.");
		}
	}

	trainSize = static_cast<size_t>(features.size() * trainRatio);

	// Shuffle the indices of the samples
	std::vector<size_t> indices(features.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::mt19937 generator(std::random_device{}());
	std::shuffle(indices.begin(), indices.end(), generator);

	// Split the features and labels into training and testing sets
	if (trainSize > features.size()) {
		trainSize = features.size();  // Adjust trainSize if it exceeds the available samples
	}

	trainData.reserve(trainSize);
	trainLabels.reserve(trainSize);
	testData.reserve(features.size() - trainSize);
	testLabels.reserve(features.size() - trainSize);

	for (size_t i = 0; i < features.size(); ++i) {
		const size_t index = indices[i];
		if (i < trainSize) {
			trainData.push_back(features[index]);
			trainLabels.push_back(labels[index]);
		}
		else {
			testData.push_back(features[index]);
			testLabels.push_back(labels[index]);
		}
	}
}



// Function to normalize dataset using mean and std deviation
void DataPreprocessor::normalizeDataset(std::vector<std::vector<double>>&dataset) {
	if (dataset.empty() || dataset[0].empty()) {
		throw std::runtime_error("Error: Dataset is empty.");
	}

	for (size_t i = 0; i < dataset[0].size(); i++) {
		std::vector<double> columnData;
		for (size_t j = 0; j < dataset.size(); j++) {
			columnData.push_back(dataset[j][i]);
		}
		double mean = std::accumulate(columnData.begin(), columnData.end(), 0.0) / columnData.size();
		double variance = 0;
		for (size_t j = 0; j < columnData.size(); j++) {
			variance += std::pow((columnData[j] - mean), 2);
		}
		if (columnData.size() > 1) {
			variance /= (columnData.size() - 1);
		}
		else {
			variance = 0.0;
		}
		double stdDeviation = std::sqrt(variance);

		// Handle the case when standard deviation is zero
		if (stdDeviation == 0.0) {
			for (size_t j = 0; j < dataset.size(); j++) {
				dataset[j][i] = 0.0;
			}
		}
		else {
			for (size_t j = 0; j < dataset.size(); j++) {
				dataset[j][i] = (dataset[j][i] - mean) / stdDeviation;
			}
		}
	}
}



// Function to scale dataset using min and max value
void DataPreprocessor::scaleDataset(std::vector<std::vector<double>>& dataset) {
	if (dataset.empty() || dataset[0].empty()) {
		throw std::runtime_error("Error: Dataset is empty.");
	}

	for (size_t i = 0; i < dataset[0].size(); i++) {
		double minVal = dataset[0][i];
		double maxVal = dataset[0][i];
		for (size_t j = 1; j < dataset.size(); j++) {
			if (dataset[j][i] < minVal) {
				minVal = dataset[j][i];
			}
			if (dataset[j][i] > maxVal) {
				maxVal = dataset[j][i];
			}
		}
		double range = maxVal - minVal;
		for (size_t j = 0; j < dataset.size(); j++) {
			if (range == 0.0) {
				dataset[j][i] = 0.0;
			}
			else {
				dataset[j][i] = (dataset[j][i] - minVal) / range;
			}
		}
	}
}


// Function to convert value to double, if not use default value
double DataPreprocessor::safeStod(const std::string& str, double default_val)
{
	try {
		return std::stod(str);
	}
	catch (const std::exception&) {
		std::cerr << "Error: Failed to convert \"" << str << "\" to double. Using default value: " << default_val << std::endl;
		return default_val;
	}
}



