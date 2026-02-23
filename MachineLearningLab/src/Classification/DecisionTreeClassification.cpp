#include "DecisionTreeClassification.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/EntropyFunctions.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <stdexcept>
#include "../DataUtils/DataPreprocessor.h"
using namespace System::Windows::Forms; // For MessageBox

// DecisionTreeClassification class implementation //


// DecisionTreeClassification is a constructor for DecisionTree class.//
DecisionTreeClassification::DecisionTreeClassification(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {}

DecisionTreeClassification::~DecisionTreeClassification() {
	clearTree(root);
	root = nullptr;
}


// Fit is a function to fits a decision tree to the given data.//
void DecisionTreeClassification::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	clearTree(root);
	root = nullptr;
	n_feats = (n_feats == 0) ? static_cast<int>(X[0].size()) : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
	if (root == nullptr) {
		throw std::logic_error("Model has not been fitted.");
	}
	std::vector<double> predictions;
	
	for (auto v : X) {
		predictions.push_back(traverseTree(v, root));
	}

	return predictions;
}

void DecisionTreeClassification::clearTree(Node* node) {
	if (node == nullptr) {
		return;
	}
	clearTree(node->left);
	clearTree(node->right);
	delete node;
}


// growTree function: This function grows a decision tree using the given data and labelsand  return a pointer to the root node of the decision tree.//
Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	
	/* Implement the following:
		--- define stopping criteria
    	--- Loop through candidate features and potential split thresholds.
		--- greedily select the best split according to information gain
		--- grow the children that result from the split
	*/
	
	// Stopping criteria
	if (depth >= max_depth || X.size() < min_samples_split || EntropyFunctions::entropy(y) == 0.0) {
		return new Node(-1, 0.0, nullptr, nullptr, mostCommonlLabel(y)); // Create a leaf node with the most common label
	}

	double best_gain = -1.0; // Initialize the best information gain
	int split_idx = -1; // Initialize the best feature to split on
	double split_thresh = 0.0; // Initialize the best threshold to split on

	// Iterate over all features
	for (size_t feature_idx = 0; feature_idx < X[0].size(); ++feature_idx) {
		// Get the column of feature values
		std::vector<double> feature_column(X.size());
		for (size_t i = 0; i < X.size(); ++i) {
			feature_column[i] = X[i][feature_idx]; 
		}

		// Find unique threshold values to split on
		std::set<double> unique_thresholds(feature_column.begin(), feature_column.end());

		// Test all thresholds for this feature
		for (const auto& threshold : unique_thresholds) {
			double gain = informationGain(y, feature_column, threshold);

			// If this split is the best so far, save it
			if (gain > best_gain) {
				best_gain = gain;
				split_idx = static_cast<int>(feature_idx);
				split_thresh = threshold;
			}
		}
	}

	// If no valid split was found, return a leaf node
	if (best_gain == -1.0) {
		return new Node(-1, 0.0, nullptr, nullptr, mostCommonlLabel(y)); // Leaf node with the most common label
	}

	// Split the data into left and right subsets
	std::vector<std::vector<double>> left_X, right_X;
	std::vector<double> left_y, right_y;
	for (size_t i = 0; i < X.size(); ++i) {
		if (X[i][split_idx] <= split_thresh) {
			left_X.push_back(X[i]);
			left_y.push_back(y[i]);
		}
		else {
			right_X.push_back(X[i]);
			right_y.push_back(y[i]);
		}
	}

	// Grow the left and right subtrees recursively
	Node* left = growTree(left_X, left_y, depth + 1);
	Node* right = growTree(right_X, right_y, depth + 1);

	// Return a new node with the split information and child nodes
	return new Node(split_idx, split_thresh, left, right);
}


/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	// parent loss // You need to caculate entropy using the EntropyFunctions class//
	double parent_entropy = EntropyFunctions::entropy(y);

	/* Implement the following:
	   --- generate split
	   --- compute the weighted avg. of the loss for the children
	   --- information gain is difference in loss before vs. after split
	*/

	// Initialize the left and right child datasets
	std::vector<double> left_y;
	std::vector<double> right_y;

	// Generate split based on the split threshold
	for (size_t i = 0; i < X_column.size(); ++i) {
		if (X_column[i] <= split_thresh) {
			left_y.push_back(y[i]);
		}
		else {
			right_y.push_back(y[i]);
		}
	}

	// If there's no valid split, return zero information gain
	if (left_y.empty() || right_y.empty()) {
		return 0.0;
	}

	// Compute the entropies for the children (left and right subsets)
	double left_entropy = EntropyFunctions::entropy(left_y);
	double right_entropy = EntropyFunctions::entropy(right_y);

	// Compute the weighted average of the children�s entropies
	double left_weight = static_cast<double>(left_y.size()) / y.size();
	double right_weight = static_cast<double>(right_y.size()) / y.size();
	double weighted_avg_entropy = left_weight * left_entropy + right_weight * right_entropy;

	// Compute the information gain (parent entropy - weighted average child entropy)
	double ig = parent_entropy - weighted_avg_entropy;

	return ig;
}


// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonlLabel(std::vector<double>& y) {	
	double most_common = 0.0;
	
	double max_count = -1.0;
	std::map<double, int> label_count;

	for (auto label : y) {
		label_count[label]++;
	}

	for (auto &p : label_count) {
		if (max_count < p.second) {
			most_common = p.first;
			max_count = p.second;
		}
	}

	return most_common;
}


// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {

	/* Implement the following:
		--- If the node is a leaf node, return its value
		--- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
		--- Otherwise, traverse the right subtree
	*/

	if (node->isLeafNode()) {
		return node->value;
	}

	if (x[node->feature] <= node->threshold) {
		return traverseTree(x,node->left);
	}

	return traverseTree(x, node->right);
}


/// runDecisionTreeClassification: this function runs the decision tree classification algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
DecisionTreeClassification::runDecisionTreeClassification(const std::string& filePath, int trainingRatio) {
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
		fit(trainData, trainLabels);//

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