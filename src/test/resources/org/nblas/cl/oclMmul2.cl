// Literatur
// http://www.fixstars.com/en/opencl/book/OpenCLProgrammingBook/opencl-c/
// http://www.informit.com/articles/article.aspx?p=1732873&seqNum=3
// https://developer.apple.com/library/mac/samplecode/OpenCL_Parallel_Reduction_Example/Listings/reduce_float4_kernel_cl.html

#define BLOCK_SIZE 16
#define BLOCK_DIM 16
#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]

// Matrices are stored in column-major order:
// M(row, col) = *(M.elements + col * M.stride + row)
typedef struct {
	int height;
	int width;
	int stride;
	__global float* elements;
} Matrix;

// Get a matrix element
float get_element(const Matrix A, int row, int col)
{
	return A.elements[col * A.stride + row];
}

// Set a matrix element
void set_element(Matrix A, int row, int col, float value)
{
	A.elements[col * A.stride + row] = value;
}

// Get the block_dim x block_dim sub-matrix Asub of A that is
// located block_col sub-matrices to the right and block_row sub-matrices down
// from the upper-left corner of A
Matrix get_sub_matrix(Matrix A, int block_row, int block_col)
{
	Matrix Asub;
	Asub.height = BLOCK_DIM;
	Asub.width = BLOCK_DIM;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_DIM * block_col + BLOCK_DIM * block_row];
	return Asub;
}

//Running CLFloatMatrixPerformanceTest with 1000 epochs
//	BlockMmulPerformanceTest:
//	Mdim:  32	     26ms
//	Mdim:  64	     21ms
//	Mdim: 128	     24ms
//	Mdim: 256	     44ms
//	Mdim: 512	    258ms
//	Mdim:1024	   1948ms
//	Mdim:2048	  15130ms
//	Mdim:4096	 146045ms
//
// Matrix multiplication function called by mmul_kernel()
void oclMmul2Impl(
		const Matrix A,
		const Matrix B,
		Matrix C,
		__const int M,
		__local float* As,
		__local float* Bs)
{
	// Block row and column
	int block_row = get_group_id(0);
	int block_col = get_group_id(1);

	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = get_sub_matrix(C, block_row, block_col);

	// Each thread computes one element of Csub
	// by accumulating results into Cvalue
	float Cvalue = 0;

	// Thread row and column within Csub
	int row = get_local_id(0);
	int col = get_local_id(1);

	// Loop over all the sub-matrices of A and B that are
	// required to compute Csub
	// Multiply each pair of sub-matrices together
	// and accumulate the results
	for (int m = 0; m < (M / BLOCK_DIM); ++m) {

		// Get sub-matrix Asub of A
		Matrix Asub = get_sub_matrix(A, block_row, m);

		// Get sub-matrix Bsub of B
		Matrix Bsub = get_sub_matrix(B, m, block_col);

		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[col * BLOCK_DIM + row] = get_element(Asub, row, col);
		Bs[col * BLOCK_DIM + row] = get_element(Bsub, row, col);

		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		barrier(CLK_LOCAL_MEM_FENCE);

		// Multiply Asub and Bsub together

		for (int e = 0; e < BLOCK_DIM; ++e) {
			Cvalue += As[e * BLOCK_DIM + row] * Bs[col * BLOCK_DIM + e];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Write Csub to device memory
	// Each thread writes one element
	set_element(Csub, row, col, Cvalue);
}


__kernel void oclMmul2(
	__global float* Aelements,
	__global float* Belements,
	__global float* Celements,
	__const int L,
	__const int M,
	__const int N,
	__local float* As,
	__local float* Bs)
{
	Matrix A = { L, M, L, Aelements };
	Matrix B = { M, N, M, Belements };
	Matrix C = { L, N, L, Celements };
	oclMmul2Impl(A, B, C, M, As, Bs);
}
