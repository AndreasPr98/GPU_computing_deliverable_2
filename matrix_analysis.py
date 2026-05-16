import os
import glob
import numpy as np
from scipy.io import mmread
from scipy.sparse import issparse

def analyze_matrix(file_path):
    try:
        # Extract the actual file name without the path and without '.mtx'
        matrix_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load the matrix (reads both coordinate and array formats)
        matrix = mmread(file_path)
        
        # Ensure it's a sparse matrix for row-based analysis
        if not issparse(matrix):
            matrix = matrix.tocsr() if hasattr(matrix, 'tocsr') else matrix
            
        # Convert to CSR (Compressed Sparse Row) for efficient row operations
        csr = matrix.tocsr()
        
        n_rows, n_cols = csr.shape
        nnz = csr.nnz
        
        # Calculate sparsity percentage
        total_elements = n_rows * n_cols
        sparsity = (1.0 - (nnz / total_elements)) * 100.0
        
        # Calculate the number of non-zero elements per row
        row_lengths = np.diff(csr.indptr)
        
        # Compute metrics
        longest_row = int(np.max(row_lengths))
        shortest_row = int(np.min(row_lengths))
        avg_row_length = float(np.mean(row_lengths))
        var_row_length = float(np.var(row_lengths))
        
        return {
            "name": matrix_name,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "sparsity": sparsity,
            "longest_row": longest_row,
            "shortest_row": shortest_row,
            "avg_row_length": avg_row_length,
            "var_row_length": var_row_length,
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_matrix_directory(root_dir, output_file="matrix_analysis_report.txt"):
    # Added 'Sparsity' column to the layout headers
    header = f"{'Matrix Name':<20} | {'Rows':<10} | {'Cols':<10} | {'NNZ':<12} | {'Sparsity':<10} | {'Max Row':<8} | {'Min Row':<8} | {'Avg Row':<10} | {'Var Row':<14}\n"
    separator = "-" * len(header) + "\n"
    
    # Open the text file for writing the results
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("SPARSE MATRIX ANALYSIS REPORT\n")
        f.write("=" * (len(header) - 1) + "\n\n")
        f.write(header)
        f.write(separator)
        
        # Also print to console to watch progress
        print(header, end="")
        print(separator, end="")
        
        # Target only the immediate subdirectories of 'Matrixes'
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            
            # Look for .mtx files strictly inside this folder (ignoring deeper nested folders)
            mtx_files = glob.glob(os.path.join(subdir_path, "*.mtx"))
            
            for mtx_file in mtx_files:
                stats = analyze_matrix(mtx_file)
                if stats:
                    # Injected the sparsity float formatted to 4 decimal places (ex: 99.9995%)
                    row_str = (f"{stats['name']:<20} | "
                               f"{stats['n_rows']:<10} | "
                               f"{stats['n_cols']:<10} | "
                               f"{stats['nnz']:<12} | "
                               f"{stats['sparsity']:<9.4f}% | "
                               f"{stats['longest_row']:<8} | "
                               f"{stats['shortest_row']:<8} | "
                               f"{stats['avg_row_length']:<10.2f} | "
                               f"{stats['var_row_length']:<14.2f}\n")
                    
                    f.write(row_str)
                    print(row_str, end="")
                    
    print(f"\nAnalysis complete! Report saved to '{output_file}'")

if __name__ == "__main__":
    root_directory = "Matrixes" 
    
    if os.path.exists(root_directory):
        process_matrix_directory(root_directory)
    else:
        print(f"Directory '{root_directory}' not found. Please check the path.")