#include "EntropyFunctions.h"
#include <vector>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>


									// EntropyFunctions class implementation //



/// Calculates the entropy of a given set of labels "y".///
double EntropyFunctions::entropy(const std::vector<double>& y) {
	size_t total_samples = y.size();
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	double entropy = 0.0;
	
	// Convert labels to unique integers and count their occurrences
	for (const auto& label : y) {
		label_map[label]++;
	}
	
	// Compute the probability and entropy
	for (const auto& pair : label_map) {
		double prop = static_cast<double>(pair.second) / total_samples;
		entropy -= prop * std::log2(prop);
	}

	return entropy;
}


/// Calculates the entropy of a given set of labels "y" and the indices of the labels "idxs".///
double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	size_t total_samples = idxs.size();
	double entropy = 0.0;

	// Convert labels to unique integers and count their occurrences based on indices
	for (const auto& idx : idxs) {
		double label = y[idx];
		label_map[label]++;
	}

	// Compute the probability and entropy
	for (const auto& pair : label_map) {
		double prob = static_cast<double>(pair.second) / total_samples;
		entropy -= prob * std::log2(prob); 
	}


	return entropy;
}


