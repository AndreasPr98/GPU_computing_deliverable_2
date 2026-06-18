1) Matrices folder
The Matrices folder needs to be structured in the following way:
- Folder Name: Matrixes
- Contains the folders of the matrices, 1 folder for each matrix
- Inside the folder there cannot be more than 1 matrix, it's okk if there is another folder.
- The single folders names dont matter, no zip, only extracted folders
- To be sure to get everything right please use the following link:
https://drive.google.com/drive/folders/14irYCCn_vZ5irZsBWZDFlcgJwK8UNmGY?usp=sharing
- If for some reason you want to use more, or less then 11 matrices be sure to change SpMV.slurm, in particular $(seq 1 11); if you want you can also use that to launch single matrix, to know what matrix will be launched just alphabetically order them.

Here there are the links for the 11 matrices to download them from the suite sparse collection:
a) https://sparse.tamu.edu/Sandia/ASIC_680ks
b) https://sparse.tamu.edu/Oberwolfach/bone010
c) https://sparse.tamu.edu/GHS_indef/boyd2
d) https://sparse.tamu.edu/Freescale/FullChip
e) https://sparse.tamu.edu/PARSEC/Ga41As41H72
f) https://sparse.tamu.edu/GHS_psdef/ldoor
g) https://sparse.tamu.edu/Williams/mc2depi
h) https://sparse.tamu.edu/Rajat/rajat31
i) https://sparse.tamu.edu/PARSEC/Si41Ge41H72
j) https://sparse.tamu.edu/Janna/StocF-1465
k) https://sparse.tamu.edu/Williams/webbase-1M


2) Matrix analysis launch 
- To use the matrix_analysis.py and get some info on the matrices you need to first load the SciPy-bundle/2023.07-gfbf-2023a or any scipy module on your HPC;
- Then you can launch it using matrix_analysis.slurm, the two files need to be in the same directory as the Matrixes folder;
- No need to change any folder paths
- For 10 matrices of ~400 MB it takes roughly 2.5 min;
- For the 11 matrices there is already a matrix_analysis_report.txt already precomputed;



3) SpMV launch
- First you need to load the CUDA/12.5.0 module
- Second you need to change the absolute path 
const std::filesystem::path base_path = "absolute path to the directory that contains GPU_code"
- Third, being in the same directory as SpMV.cu (aka the main) you can press make and that the makefile compile the program
- After the makefile has finished you can lauch a job with "sbatch SpMV.slurm" 
- After completion the data will be inside the directory "GPU_code/program_output", here you can extract the csv file, or directly open the .out file to vision the output in a more structured way.
- A basic csv and an elaborated excel file are already present as an output example