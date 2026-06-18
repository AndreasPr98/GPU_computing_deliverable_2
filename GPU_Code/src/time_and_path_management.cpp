#include "time_and_path_management.h"
#include <iostream>
#include <algorithm>

bool handle_path_management(int argc, char* argv[], 
                            const std::filesystem::path& base_path, 
                            const std::filesystem::path& target_dir, 
                            std::filesystem::directory_entry& out_entry) {

    // 1. Check if the matrices directory exists
    if (!std::filesystem::exists(target_dir) || !std::filesystem::is_directory(target_dir)) {
        std::cerr << "Error: Invalid directory path: " << target_dir << std::endl;
        return false;
    }
    
    // 2. Collect matrix files
    std::vector<std::filesystem::path> matrix_files;
    for (auto it = std::filesystem::recursive_directory_iterator(target_dir);
         it != std::filesystem::recursive_directory_iterator(); ++it) {
        
        // Limit depth to 1 level
        if (it.depth() > 1) { 
            it.disable_recursion_pending(); 
            continue; 
        }

        const auto& entry = *it;
        if (entry.is_regular_file() && entry.path().extension() == ".mtx") {
            matrix_files.push_back(entry.path());
        }
    }

    // 3. Sort the matrices in alphabetical order
    std::sort(matrix_files.begin(), matrix_files.end(),
        [](const std::filesystem::path& a, const std::filesystem::path& b) {
            return a.filename().string() < b.filename().string();
        });

    // 4. Check if the matrix number has been passed via CLI arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_number 1-" << matrix_files.size() << ">" << std::endl;
        return false;
    }

    // 5. Check if the data passed through argv is actually a number
    char* endptr = nullptr;
    long requested_matrix_number = strtol(argv[1], &endptr, 10);
    if (endptr == argv[1] || *endptr != '\0') {
        std::cerr << "Invalid matrix number: " << argv[1] << std::endl;
        return false;
    }

    // 6. Check if the number is in range of 1 - total_n_of_matrixes
    if (requested_matrix_number < 1 || (size_t)requested_matrix_number > matrix_files.size()) {
        std::cerr << "Matrix number must be in range [1, " << matrix_files.size() << "]." << std::endl;
        return false;
    }
    
    // Get the selected matrix from the sorted list 
    out_entry = std::filesystem::directory_entry(matrix_files[(size_t)requested_matrix_number - 1]);

    return true;
}
