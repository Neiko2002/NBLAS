// Settings
#define RX 8
#define RY 4
#define RK (RY)

// Mimic clBlas (4x8 register tiling with vector data-types)
__kernel void mySGEMM11(
                       const __global float8* restrict A,
                       const __global float4* restrict B,
                       __global float8* C,
					   const int M, const int N, const int K) {

    // Allocate register space
    float aReg[RK][RX];
    float bReg[RY][RK];
    float acc[RY][RX];

    // Initialise the accumulation registers
    for (int y=0; y<RY; y++) {
        for (int x=0; x<RX; x++) {
            acc[y][x] = 0.0;
        }
    }

    // Loop over all tiles
    const int numTiles = K/RK;
    for (int t=0; t<numTiles; t++) {

        // Load a tile of A and B into register memory
        for (int y=0; y<RY; y++) {

            // Load the data
            float8 aVec = A[(RK*t + y)*(M/RX) + get_global_id(0)];
            float4 bVec = B[(RY*get_global_id(1) + y)*numTiles + t];

            // Store the vector of A into registers
            aReg[y][0] = aVec.s0;
            aReg[y][1] = aVec.s1;
            aReg[y][2] = aVec.s2;
            aReg[y][3] = aVec.s3;
            aReg[y][4] = aVec.s4;
            aReg[y][5] = aVec.s5;
            aReg[y][6] = aVec.s6;
            aReg[y][7] = aVec.s7;

            // Store the vector of B into registers
            bReg[y][0] = bVec.x;
            bReg[y][1] = bVec.y;
            bReg[y][2] = bVec.z;
            bReg[y][3] = bVec.w;
        }

        // Perform the computations
        for (int k=0; k<RK; k++) {
            for (int y=0; y<RY; y++) {
                for (int x=0; x<RX; x++) {
                    acc[y][x] += aReg[k][x] * bReg[y][k];
                }
            }
        }
    }

    // Store the final results in C
    for (int y=0; y<RY; y++) {
        float8 accVec;
        accVec.s0 = acc[y][0];
        accVec.s1 = acc[y][1];
        accVec.s2 = acc[y][2];
        accVec.s3 = acc[y][3];
        accVec.s4 = acc[y][4];
        accVec.s5 = acc[y][5];
        accVec.s6 = acc[y][6];
        accVec.s7 = acc[y][7];
        C[(y + RY*get_global_id(1))*(M/RX) + get_global_id(0)] = accVec;
    }
}
