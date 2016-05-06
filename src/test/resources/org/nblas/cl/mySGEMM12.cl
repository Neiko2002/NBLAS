typedef union GPtr {
    __global float *f;
    __global float2 *f2v;
    __global float4 *f4v;
    __global float8 *f8v;
    __global float16 *f16v;
} GPtr;

__attribute__((reqd_work_group_size(8, 8, 1)))
void __kernel mySGEMM12(
		const __global float8 *restrict A,
		const __global float4 *restrict B,
		__global float8 *C,
		uint M,
		uint N,
		uint K,
		const float alpha,
		const float beta,
		uint lda,
		uint ldb,
		uint ldc)
{
    float8 a0, a1, a2, a3;
    float4 b0, b1, b2, b3;
    float8 c0, c1, c2, c3;
    uint4 coord = 0u; /* contains coordB, coordA, k */

    lda /= 8;
    ldb /= 4;
    A += (uint)get_global_id(0);
    B += 4u * (uint)get_global_id(1) * ldb;

    coord.y = 8u * (uint)get_global_id(0);
    coord.x = 4u * (uint)get_global_id(1);
    c0 = 0;
    c1 = 0;
    c2 = 0;
    c3 = 0;

    for (uint k1 = 0; k1 < K; k1 += 4) {
        /* -- Tiles multiplier -- */
        b0 = B[0];
        b1 = B[ldb];
        b2 = B[(ldb << 1)];
        b3 = B[mad24(3u, ldb, 0u)];

        a0 = A[0];
        a1 = A[lda];
        a2 = A[(lda << 1)];
        a3 = A[mad24(3u, lda, 0u)];

        c0 += a0 * b0.s0;
        c1 += a0 * b1.s0;
        c2 += a0 * b2.s0;
        c3 += a0 * b3.s0;

        c0 += a1 * b0.s1;
        c1 += a1 * b1.s1;
        c2 += a1 * b2.s1;
        c3 += a1 * b3.s1;

        c0 += a2 * b0.s2;
        c1 += a2 * b1.s2;
        c2 += a2 * b2.s2;
        c3 += a2 * b3.s2;

        c0 += a3 * b0.s3;
        c1 += a3 * b1.s3;
        c2 += a3 * b2.s3;
        c3 += a3 * b3.s3;

        A += (lda << 2);
        B += 1;
        /* ---------------------- */
    }


    GPtr uC;

    uC.f = C + (coord.x * ldc + coord.y)/8;

    __global float8 *pC = uC.f8v;

    float8 tempC0, tempC1, tempC2, tempC3;

    tempC0 = c0 * alpha + 0;
    tempC1 = c1 * alpha + 0;
    tempC2 = c2 * alpha + 0;
    tempC3 = c3 * alpha + 0;
    pC[0] = tempC0;
    pC[(ldc >> 3)] = tempC1;
    pC[(ldc >> 2)] = tempC2;
    pC[mad24(3u, (ldc >> 3), 0u)] = tempC3;
}
