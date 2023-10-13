# A-transfer-framework-based-on-co-matrix-decomposition
I have completed co-matrix decomposition for undirected, high-dimensional and sparse networks with python3.8+cuda.
A transfer framework based on co-matrix decomposition for undirected, high-dimensional and sparse networks
https://doi.org/10.1016/j.jocs.2022.101677
Highlights
•
A symmetric and non-negative collective factorization is proposed for SHiDS networks.

•
To solve it, we design the collective batch-based non-negative optimization approach.

•
An auxiliary matrix construction method is designed to simulate the real scene.


Abstract
Undirected, high-dimensional and sparse network is often encountered in industrial engineering and described by symmetric, high-dimensional and sparse (SHiDS) matrix for academic investigation. Latent factor (LF) models have shown an arguably promising yet whelming capability of extracting knowledge from so little known information in the SHiDS matrix. However, most of the LF models in the form of matrix factorization (a.k.a. matrix decomposition) focus exclusively on a single domain and struggle to meet the requirements of symmetry and non-negativity of the SHiDS matrix. To address these issues, a symmetric and non-negative collective factorization (SNCF) transfer framework in the form of collective-matrix (co-matrix) decomposition is proposed. SNCF transfers a great deal of knowledge from the auxiliary domain to mitigate the sparsity of the target domain. To solve the SNCF efficiently and guarantee the symmetry and non-negativity of the target matrix, a collective batch-based non-negative optimization approach (SNCF-CBN) is put forward. Experimental results demonstrate the effectiveness of SNCF-CBN by comparing it with several state-of-the-art models on public datasets, including Protein Networks, Sonar Mines and Sonar Rocks datasets with different sparsities in cross-domain.
