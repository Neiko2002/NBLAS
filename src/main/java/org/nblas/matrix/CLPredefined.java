package org.nblas.matrix;

import java.util.HashMap;

/**
 * Created by Moritz on 7/7/2015.
 */
public class CLPredefined {

    private static final String uniform = "int xorshift(__global uint4* vec)\n" +
            "{\n" +
            "    uint t = vec[0].x ^ (vec[0].x << 11);\n" +
            "    *vec = vec[0].yzww;\n" +
            "    vec[0].w = vec[0].w ^ (vec[0].w >> 19) ^ (t ^ (t >> 8));\n" +
            "    return vec[0].w;\n" +
            "}\n" +
            "\n" +
            "__kernel void auniform(__global uint4* random, __global float* values, const uint stride, const uint rows, const uint columns)\n" +
            " {\n" +
            "     uint gid0 = get_global_id(0);\n" +
            "     uint gid1 = get_global_id(1);\n" +
            "     uint gridSize0 = get_global_size(0);\n" +
            "     uint gridSize1 = get_global_size(1);\n" +
            "     uint tid = gid1 * gridSize1 + gid0;\n" +
            "     while(gid1 < columns)\n" +
            "     {\n" +
            "         gid0 = get_global_id(0);\n" +
            "         while(gid0 < rows)\n" +
            "         {\n" +
            "             uint4 rand =\n" +
            "             values[gid1 * stride + gid0] =  (uint)xorshift(&random[tid]) / 4294967296.0f;\n" +
            "             gid0 += gridSize0;\n" +
            "         }\n" +
            "         gid1 += gridSize1;\n" +
            "     }\n" +
            " }";

    private static final String boxmuller = "__kernel void boxmuller(__global uint4* random, __global float* values, const uint stride, const uint rows, const uint columns)\n" +
            "{\n" +
            "    uint gid0 = get_global_id(0);\n" +
            "    uint gid1 = get_global_id(1);\n" +
            "    uint gridSize0 = get_global_size(0);\n" +
            "    uint gridSize1 = get_global_size(1);\n" +
            "    uint tid = gid1 * gridSize1 + gid0;\n" +
            "     while(gid1 < columns)\n" +
            "     {\n" +
            "         gid0 = get_global_id(0);\n" +
            "         while(gid0 < rows)\n" +
            "         {\n" +
            "            float random1 = (uint)xorshift(&random[tid]) / 4294967296.0f;\n" +
            "            if(random1 < 1e-10f) random1 = 1e-10f;\n" +
            "            float random2 = (uint)xorshift(&random[tid]) / 4294967296.0f;\n" +
            "\n" +
            "            random1 = -2.0f * log(random1);\n" +
            "            random2 *= 6.28318530718f;\n" +
            "\n" +
            "            values[gid1 * stride + gid0] = sqrt(random1) * cos(random2);\n" +
            "            gid0 += gridSize0;\n" +
            "         }\n" +
            "         gid1 += gridSize1;\n" +
            "     }\n" +
            "}";

    private static final String[] reductionFloats = {
            "__kernel void ",
            "(__global float* inputData, __global float* outputData, __local float* shared, const uint rows, const uint columns, const float initValue)\n" +
                    "{\n" +
                    "    int tid0 = get_local_id(0);\n" +
                    "    int tid1 = get_local_id(1);\n" +
                    "    int gid0 = get_global_id(0);\n" +
                    "    int gid1 = get_global_id(1);\n" +
                    "    int gridSize0 = get_global_size(0);\n" +
                    "    int gridSize1 = get_global_size(1);\n" +
                    "    int localSize0 = get_local_size(0);\n" +
                    "    int sIndex = tid1 * localSize0 + tid0;\n" +
                    "    shared[sIndex] = initValue;\n" +
                    "    while (gid1 < columns) {\n" +
                    "        gid0 = get_global_id(0);\n" +
                    "        while (gid0 < rows) {\n",

            "            gid0 += gridSize0;\n" +
                    "        }\n" +
                    "        gid1 += gridSize1;\n" +
                    "    }\n" +
                    " \tbarrier(CLK_LOCAL_MEM_FENCE);\n" +
                    "\n" +
                    "    gid0 = get_global_id(0);\n" +
                    "    gid1 = get_global_id(1);\n" +
                    "\n" +
                    "    for (uint s = get_local_size(1) * get_local_size(0) >> 1; s > 0; s >>= 1)\n" +
                    "    {\n" +
                    "        if (tid1 * get_local_size(0) + tid0 < s && gid1 < columns && gid0 < rows)\n",

            "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
                    "    }\n" +
                    "\n" +
                    "    if (sIndex == 0)\n" +
                    "      outputData[get_group_id(1) * get_local_size(0) + get_group_id(0)] = shared[0];\n" +
                    "}"
    };

    private static final String[] reductionColumnFloats = {
            "__kernel void ",

            "(__global float* inputData, __global float* outputData, __local float* shared, const int rows, const int columns, const float initValue)\n" +
                    "{\n" +
                    "\n" +
                    "    int tid0 = get_local_id(0);\n" +
                    "    int tid1 = get_local_id(1);\n" +
                    "    int gid0 = get_global_id(0);\n" +
                    "    int gid1 = get_global_id(1);\n" +
                    "    int gridSize0 = get_global_size(0);\n" +
                    "    if(gid1 >= columns) return;\n" +
                    "    int sIndex = tid1 * get_local_size(0) + tid0;\n" +
                    "    shared[sIndex] = initValue;\n" +
                    "    while(gid0 < rows)\n" +
                    "    {\n",

            "        gid0 += gridSize0;\n" +
                    "    }\n" +
                    " \tbarrier(CLK_LOCAL_MEM_FENCE);\n" +
                    "\n" +
                    "    gid0 = get_global_id(0);\n" +
                    "    for(unsigned int s = get_local_size(0) >> 1; s > 0; s >>= 1) {\n" +
                    "        if(tid0 < s && gid0 < rows && gid1 < columns) {\n" +
                    "            int id = tid1 * get_local_size(0) + tid0;\n",

            " \t\t\tbarrier(CLK_LOCAL_MEM_FENCE);\n" +
                    "        }\n" +
                    "    }\n" +
                    "\n" +
                    "    if(tid0 == 0)\n" +
                    "        outputData[gid1 * get_num_groups(0) + get_group_id(0)] = shared[tid1 * get_local_size(0)];\n" +
                    "}"};

    private static final String[] reductionRowFloats = {
            "__kernel void ",

            "(__global float* inputData,__global  float* outputData, __local float* shared, const int rows, const int columns, const float initValue)\n" +
                    "{\n" +
                    "    int tid0 = get_local_id(0);\n" +
                    "    int tid1 = get_local_id(1);\n" +
                    "    int gid0 = get_global_id(0);\n" +
                    "    int gid1 = get_global_id(1);\n" +
                    "    int gridSize1 = get_global_size(1);\n" +
                    "    if(gid0 >= rows) return;\n" +
                    "\n" +
                    "    int sIndex = tid1 * get_local_size(0) + tid0;\n" +
                    "        shared[sIndex] = initValue;\n" +
                    "    while(gid1 < columns)\n" +
                    "    {\n",

            "        gid1 += gridSize1;\n" +
                    "    }\n" +
                    " \tbarrier(CLK_LOCAL_MEM_FENCE);\n" +
                    "\n" +
                    "    gid1 = get_global_id(1);\n" +
                    "    for(unsigned int s = get_local_size(1) >> 1; s > 0; s >>= 1) {\n" +
                    "        if(tid1 < s && gid0 < rows && gid1 < columns) {\n",

            " \t\t\tbarrier(CLK_LOCAL_MEM_FENCE);\n" +
                    "        }\n" +
                    "    }\n" +
                    "    if(tid1 == 0)\n" +
                    "        outputData[get_group_id(1) * get_global_size(0) + gid0] = shared[tid0];\n" +
                    "}"};

    private static final String copy1D = "__kernel void copy1D(__global float* input, __global float* output, uint n)\n" +
            "{\n" +
            "   int gid = get_global_id(0);\n" +
            "   if(gid >= n) return;\n" +
            "   output[gid] = input[gid];\n" +
            "}\n";

    private static final String transpose = "__kernel void transpose(const __global float* input,\n" +
            "                        __global float* output, __local float* buffer, int rows, int columns) {\n" +
            "\n" +
            "    const int tx = get_local_id(0);\n" +
            "    const int ty = get_local_id(1);\n" +
            "    const int gid0 = get_global_id(0);\n" +
            "    const int gid1 = get_global_id(1);\n" +
            "\n" +
            "    buffer[ty * get_local_size(0) + tx] = input[gid1 * get_global_size(0) + gid0];\n" +
            "    barrier(CLK_LOCAL_MEM_FENCE);\n" +
            "\n" +
            "    const int gid0t = get_group_id(1) * get_local_size(1) + tx;\n" +
            "    const int gid1t = get_group_id(0) * get_local_size(0) + ty;\n" +
            "    output[gid1t * get_global_size(1) + gid0t] = buffer[tx * get_local_size(1) + ty];\n" +
            "}";

    private static final String setZero = "__kernel void setZero(__global float* input)\n" +
            "{\n" +
            "    const int gid0 = get_global_id(0);\n" +
            "    const int gid1 = get_global_id(1);\n" +
            "\n" +
            "    input[gid1 * get_global_size(0) + gid0] = 0.0f;\n" +
            "}";

    private static final String sgemm_nn = "__kernel void sgemm_nn(const __global float* a, const __global float* b, __global float* c,\n" +
            "                      __local float* aSub, __local float* bSub,\n" +
            "                       const int M, const int N, const int K) {\n" +
            "\n" +
            "    const int tid0 = get_local_id(0);\n" +
            "    const int tid1 = get_local_id(1);\n" +
            "    const int gid0 = get_global_id(0);\n" +
            "    const int gid1 = get_global_id(1);\n" +
            "    int localSize0 = get_local_size(0);\n" +
            "    int localSize1 = get_local_size(1);\n" +
            "\n" +
            "    float result = 0.0f;\n" +
            "\n" +
            "    int numTiles = K / localSize0;\n" +
            "    int index = tid1 * localSize0 + tid0;\n" +
            "    for (int t=0; t < numTiles; t++) {\n" +
            "\n" +
            "        int tiled0 = localSize0 * t + tid0;\n" +
            "        int tiled1 = localSize1 * t + tid1;\n" +
            "        aSub[index] = a[tiled1 * M + gid0];\n" +
            "        bSub[index] = b[gid1 * K + tiled0];\n" +
            "\n" +
            "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
            "\n" +
            "        for (int k = 0; k < localSize0; k++) {\n" +
            "            result += aSub[k * localSize0 + tid0] * bSub[tid1 * localSize0 + k];\n" +
            "        }\n" +
            "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
            "    }\n" +
            "\n" +
            "    c[gid1 * M + gid0] = result;\n" +
            "}";


    private static final String sgemm_nt = "__kernel void sgemm_nt(const __global float* a, const __global float* b, __global float* c,\n" +
            "                      __local float* aSub, __local float* bSub,\n" +
            "                       const int M, const int N, const int K) {\n" +
            "\n" +
            "    const int tid0 = get_local_id(0);\n" +
            "    const int tid1 = get_local_id(1);\n" +
            "    const int gid0 = get_global_id(0);\n" +
            "    const int gid1 = get_global_id(1);\n" +
            "    int localSize0 = get_local_size(0);\n" +
            "    int localSize1 = get_local_size(1);\n" +
            "\n" +
            "    float result = 0.0f;\n" +
            "\n" +
            "    int numTiles = K / localSize0;\n" +
            "    int indexA = tid1 * localSize0 + tid0;\n" +
            "    int indexB = tid0 * localSize1 + tid1;\n" +
            "    for (int t = 0; t < numTiles; t++) {\n" +
            "\n" +
            "//        int tiled0 = localSize0 * t + tid0;\n" +
            "        int tiled1 = localSize1 * t + tid1;\n" +
            "        aSub[indexA] = a[tiled1 * M + gid0];\n" +
            "        bSub[indexB] = b[tiled1 * N + get_group_id(1) * localSize1 + tid0];\n" +
            "\n" +
            "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
            "\n" +
            "        for (int k = 0; k < localSize0; k++) {\n" +
            "            result += aSub[k * localSize0 + tid0] * bSub[tid1 * localSize0 + k];\n" +
            "        }\n" +
            "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
            "    }\n" +
            "\n" +
            "    c[gid1 * M + gid0] = result;\n" +
            "}";

    private static final String sgemm_tn = "__kernel void sgemm_tn(const __global float* a, const __global float* b, __global float* c,\n" +
            "                      __local float* aSub, __local float* bSub,\n" +
            "                       const int M, const int N, const int K) {\n" +
            "\n" +
            "    const int tid0 = get_local_id(0);\n" +
            "    const int tid1 = get_local_id(1);\n" +
            "    const int gid0 = get_global_id(0);\n" +
            "    const int gid1 = get_global_id(1);\n" +
            "    int localSize0 = get_local_size(0);\n" +
            "    int localSize1 = get_local_size(1);\n" +
            "\n" +
            "    float result = 0.0f;\n" +
            "\n" +
            "    int numTiles = K / localSize0;\n" +
            "\n" +
            "    int indexA = tid1 * localSize0 + tid0;\n" +
            "    int indexB = tid0 * localSize1 + tid1;\n" +
            "    for (int t=0; t < numTiles; t++) {\n" +
            "\n" +
            "        int tiled0 = localSize0 * t + tid0;\n" +
            "        aSub[indexB] = a[(get_group_id(0) * localSize0 + tid1) * K + tiled0];\n" +
            "        bSub[indexA] = b[gid1 * K + tiled0];\n" +
            "\n" +
            "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
            "\n" +
            "        for (int k = 0; k < localSize0; k++) {\n" +
            "            result += aSub[k * localSize0 + tid0] * bSub[tid1 * localSize0 + k];\n" +
            "        }\n" +
            "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
            "    }\n" +
            "\n" +
            "    c[gid1 * M + gid0] = result;\n" +
            "}";

    public static final HashMap<String, String> kernels;

    static {
        kernels = new HashMap<>();
        kernels.put("auniform", uniform);
        kernels.put("boxmuller", boxmuller);
        kernels.put("copy1D", copy1D);
        kernels.put("transpose", transpose);
        kernels.put("setZero", setZero);
        kernels.put("sgemm_nn", sgemm_nn);
        kernels.put("sgemm_nt", sgemm_nt);
        kernels.put("sgemm_tn", sgemm_tn);
        String[] sum = {"sumFloats",
                "\t\tshared[sIndex] += inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\tshared[sIndex] += shared[sIndex + s];\n"};
        kernels.put(sum[0], buildReductionKernel(sum, reductionFloats));

        String[] product = {"productFloats",
                "\t\tshared[sIndex] *= inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\tshared[sIndex] *= shared[sIndex + s];\n"};
        kernels.put(product[0], buildReductionKernel(product, reductionFloats));

        String[] max = {"maxFloats",
                "\t\tshared[sIndex] = max(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tshared[sIndex] = max(shared[sIndex], shared[sIndex + s]);\n"};
        kernels.put(max[0], buildReductionKernel(max, reductionFloats));

        String[] min = {"minFloats",
                "\t\tshared[sIndex] = min(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tshared[sIndex] = min(shared[sIndex], shared[sIndex + s]);\n"};
        kernels.put(min[0], buildReductionKernel(min, reductionFloats));

        String[] columnSums = {"columnSumsFloats",
                "\t\tshared[sIndex] += inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\t\tshared[id] += shared[id + s];\n"};
        kernels.put(columnSums[0], buildReductionKernel(columnSums, reductionColumnFloats));

        String[] columnProducts = {"columnProductsFloats",
                "\t\tshared[sIndex] *= inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\t\tshared[id] *= shared[id + s];\n"};
        kernels.put(columnProducts[0], buildReductionKernel(columnProducts, reductionColumnFloats));

        String[] columnMaxs = {"columnMaxsFloats",
                "\t\tshared[sIndex] = max(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tshared[id] = max(shared[id], shared[id + s]);\n"};
        kernels.put(columnMaxs[0], buildReductionKernel(columnMaxs, reductionColumnFloats));

        String[] columnMins = {"columnMinsFloats",
                "\t\tshared[sIndex] = min(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tshared[id] = min(shared[id], shared[id + s]);\n"};
        kernels.put(columnMins[0], buildReductionKernel(columnMins, reductionColumnFloats));

        String[] rowSums = {"rowSumsFloats",
                "\t\tshared[sIndex] += inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\t\tshared[tid1 * get_local_size(0) + tid0] += shared[(tid1 + s) * get_local_size(0) + tid0];\n"
        };
        kernels.put(rowSums[0], buildReductionKernel(rowSums, reductionRowFloats));

        String[] rowProducts = {"rowProductsFloats",
                "\t\tshared[sIndex] *= inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\t\tshared[tid1 * get_local_size(0) + tid0] *= shared[(tid1 + s) * get_local_size(0) + tid0];\n"
        };
        kernels.put(rowProducts[0], buildReductionKernel(rowProducts, reductionRowFloats));

        String[] rowMaxs = {"rowMaxsFloats",
                "\t\tshared[sIndex] = max(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tunsigned int sharedIndex = tid1 * get_local_size(0) + tid0;\n" +
                        "\t\t\tshared[sharedIndex] = max(shared[sharedIndex], shared[(tid1 + s) * get_local_size(0) + tid0]);\n"
        };
        kernels.put(rowMaxs[0], buildReductionKernel(rowMaxs, reductionRowFloats));

        String[] rowMins = {"rowMinsFloats",
                "\t\tshared[sIndex] = min(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tunsigned int sharedIndex = tid1 * get_local_size(0) + tid0;\n" +
                        "\t\t\tshared[sharedIndex] = min(shared[sharedIndex], shared[(tid1 + s) * get_local_size(0) + tid0]);\n"
        };
        kernels.put(rowMins[0], buildReductionKernel(rowMins, reductionRowFloats));

    }


    private static String buildReductionKernel(String[] function, String[] reduction) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < reduction.length - 1; i++) {
            builder.append(reduction[i]);
            builder.append(function[i]);
        }
        builder.append(reduction[reduction.length - 1]);
        return builder.toString();
    }
}
