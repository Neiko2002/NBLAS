//Running CLFloatMatrixPerformanceTest with 1000 epochs
//	BlockMmulPerformanceTest:
//	Mdim:  32	     25ms
//	Mdim:  64	     19ms
//	Mdim: 128	     21ms
//	Mdim: 256	     53ms
//	Mdim: 512	    318ms
//	Mdim:1024	   2412ms
//	Mdim:2048	  18843ms
//	Mdim:4096	 152794ms


#define BLOCK_SIZE 32
#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]


__kernel void oclMmul1(
	__global const float* A,
	__global const float* B,
	__global float* C,
	__local float* As,
	__local float* Bs,
	const int aColumns, const int bColumns)
{
	// Block index
	int bx = get_group_id(0);
	int by = get_group_id(1);

	// Thread index
	int tx = get_local_id(0);
 	int ty = get_local_id(1);

	// Index of the first sub-matrix of A processed by the block
	int aBegin = aColumns * BLOCK_SIZE * by + aColumns * ty + tx;

	// Index of the last sub-matrix of A processed by the block
	int aEnd   = aBegin + aColumns;

	// Step size used to iterate through the sub-matrices of A
	int aStep  = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx + bColumns * ty + tx;

	// Step size used to iterate through the sub-matrices of B
 	int bStep  = BLOCK_SIZE * bColumns;

	// total is used to store the element of the block sub-matrix
	// that is computed by the thread
	float total = 0.0f;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a < aEnd; a += aStep, b += bStep) {

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		AS(ty, tx) = A[a];
		BS(ty, tx) = B[b];

		// Synchronize to make sure the matrices are loaded
		barrier(CLK_LOCAL_MEM_FENCE);

 		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
		#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
			total += AS(ty, k) * BS(k, tx);

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = total;
 }
