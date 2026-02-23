#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>


class DataLoader {
public:
    static void loadAndPreprocessDataset(const std::string& file_path, std::vector<std::vector<double>>& dataset);
    static std::vector<std::vector<std::string>> readDatasetFromFilePath(const std::string& filePath);

  
};

#endif // DATALOADER_H


