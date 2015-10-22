package org.nblas.matrix;

import java.util.HashMap;

/**
 * Created by Moritz on 6/1/2015.
 */
class CudaPredefined {

    public static final HashMap<String, String> kernels;


    private final static String copy1D = "extern \"C\"\n" +
            "__global__ void copy1D(float* inputData, float* outputData, const int n)\n" +
            "{\n" +
            "    unsigned int id0 = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    if(id0 >= n) return;\n" +
            "    outputData[id0] = inputData[id0];\n" +
            "\n" +
            "}";

    private static final String[] reductionFloats = {
            "extern \"C\"\n" +
                    "__global__ void ",

            "(float* inputData, float* outputData, const int elements, const float initValue)\n" +
                    "{\n" +
                    "    extern __shared__ float shared[];\n" +
                    "\n" +
                    "    int tid = threadIdx.x;\n" +
                    "    int gid = blockDim.x * blockIdx.x + tid;\n" +
                    "    int gridsize = gridDim.x * blockDim.x;\n" +
                    "    shared[tid] = initValue;\n" +
                    "    while (gid < elements) {\n",

            "        gid += gridsize;\n" +
                    "    }\n" +
                    "    __syncthreads();\n" +
                    "\n" +
                    "    gid = blockDim.x * blockIdx.x + tid;\n" +
                    "    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)\n" +
                    "    {\n" +
                    "        if (tid < s && gid < elements)\n",

            "        __syncthreads();\n" +
                    "    }\n" +
                    "\n" +
                    "    if (tid == 0)\n" +
                    "      outputData[blockIdx.x] = shared[0];\n" +
                    "}"
    };

    private static final String[] reductionColumnFloats = {
            "extern \"C\"\n" +
                    "__global__ void ",

            "(float* inputData, float* outputData,const int rows, const int columns, const float initValue)\n" +
                    "{\n" +
                    "    extern __shared__ float shared[];\n" +
                    "\n" +
                    "    int tid0 = threadIdx.x;\n" +
                    "    int tid1 = threadIdx.y;\n" +
                    "    int gid0 = blockDim.x * blockIdx.x + tid0;\n" +
                    "    int gid1 = blockDim.y * blockIdx.y + tid1;\n" +
                    "    int gridSize0 = gridDim.x * blockDim.x;\n" +
                    "    if(gid1 >= columns) return;\n" +
                    "    int sIndex = tid1 * blockDim.x + tid0;\n" +
                    "    shared[sIndex] = initValue;\n" +
                    "    while(gid0 < rows)\n" +
                    "    {\n",

            "        gid0 += gridSize0;\n" +
                    "    }\n" +
                    "    __syncthreads();\n" +
                    "\n" +
                    "    gid0 = blockDim.x * blockIdx.x + tid0;\n" +
                    "    for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {\n" +
                    "        if(tid0 < s && gid0 < rows && gid1 < columns) {\n" +
                    "            int id = tid1 * blockDim.x + tid0;\n",

            "            __syncthreads();\n" +
                    "        }\n" +
                    "    }\n" +
                    "\n" +
                    "    if(tid0 == 0)\n" +
                    "        outputData[gid1 * gridDim.x + blockIdx.x] = shared[tid1 * blockDim.x];\n" +
                    "}"
    };

    private static final String[] reductionRowFloats = {
            "extern \"C\"\n" +
                    "__global__ void ",
            "(float* inputData, float* outputData,const int rows, const int columns, const float initValue)\n" +
                    "{\n" +
                    "    extern __shared__ float shared[];\n" +
                    "\n" +
                    "    int tid0 = threadIdx.x;\n" +
                    "    int tid1 = threadIdx.y;\n" +
                    "    int gid0 = blockDim.x * blockIdx.x + tid0;\n" +
                    "    int gid1 = blockDim.y * blockIdx.y + tid1;\n" +
                    "    int gridSize1 = gridDim.y * blockDim.y;\n" +
                    "    if(gid0 >= rows) return;\n" +
                    "\n" +
                    "    int sIndex = tid1 * blockDim.x + tid0;\n" +
                    "        shared[sIndex] = initValue;\n" +
                    "    while(gid1 < columns)\n" +
                    "    {\n",

            "        gid1 += gridSize1;\n" +
                    "    }\n" +
                    "    __syncthreads();\n" +
                    "\n" +
                    "    gid1 = blockDim.y * blockIdx.y + tid1;\n" +
                    "    for(unsigned int s = blockDim.y >> 1; s > 0; s >>= 1) {\n" +
                    "        if(tid1 < s && gid0 < rows && gid1 < columns) {\n",

            "            __syncthreads();\n" +
                    "        }\n" +
                    "    }\n" +
                    "    if(tid1 == 0)\n" +
                    "        outputData[blockIdx.y * rows + gid0] = shared[tid0];\n" +
                    "}"
    };

    private static final String transpose = "extern \"C\"\n" +
            "__global__ void transpose(float* inputData, float* outputData, const int rows, const int columns)\n" +
            "{\n" +
            "\n" +
            "    int tid0 = threadIdx.x;\n" +
            "    int tid1 = threadIdx.y;\n" +
            "    int gid0 = blockDim.x * blockIdx.x + tid0;\n" +
            "    int gid1 = blockDim.y * blockIdx.y + tid1;\n" +
            "\n" +
            "    extern __shared__ float shared[];\n" +
            "    shared[tid1 * blockDim.x + tid0] = inputData[gid1 * rows + gid0];\n" +
            "    __syncthreads();\n" +
            "\n" +
            "    int gid0t = blockDim.y * blockIdx.y + tid0;\n" +
            "    int gid1t = blockDim.x * blockIdx.x + tid1;\n" +
            "\n" +
            "\n" +
            "    if(gid0t >= columns || gid1t >= rows) return;\n" +
            "    outputData[gid1t * columns + gid0t] = shared[tid0 * blockDim.y + tid1];\n" +
            "}";

    static {
        kernels = new HashMap<>();
        kernels.put("copy1D", copy1D);
        kernels.put("transpose", transpose);
        String[] sum = {"sumFloats",
                "\t\tshared[tid] += inputData[gid];\n",
                "\t\tshared[tid] += shared[tid + s];\n"};
        kernels.put(sum[0], getFunction(sum, reductionFloats));

        String[] product = {"productFloats",
                "\t\tshared[tid] *= inputData[gid];\n",
                "\t\tshared[tid] *= shared[tid + s];\n"};
        kernels.put(product[0], getFunction(product, reductionFloats));

        String[] max = {"maxFloats",
                "\t\tshared[tid] = max(shared[tid], inputData[gid]);\n",
                "\t\tshared[tid] = max(shared[tid], shared[tid + s]);\n"};
        kernels.put(max[0], getFunction(max, reductionFloats));

        String[] min = {"minFloats",
                "\t\tshared[tid] = min(shared[tid], inputData[gid]);\n",
                "\t\tshared[tid] = min(shared[tid], shared[tid + s]);\n"};
        kernels.put(min[0], getFunction(min, reductionFloats));

        String[] columnSums = {"columnSumsFloats",
                "\t\tshared[sIndex] += inputData[gid1 * rows + gid0];\n",
                "\t\t\tshared[id] += shared[id + s];\n"};
        kernels.put(columnSums[0], getFunction(columnSums, reductionColumnFloats));

        String[] columnProducts = {"columnProductsFloats",
                "\t\tshared[sIndex] *= inputData[gid1 * rows + gid0];\n",
                "\t\t\tshared[id] *= shared[id + s];\n"};
        kernels.put(columnProducts[0], getFunction(columnProducts, reductionColumnFloats));

        String[] columnMaxs = {"columnMaxsFloats",
                "\t\tshared[sIndex] = max(shared[sIndex], inputData[gid1 * rows + gid0]);\n",
                "\t\t\tshared[id] = max(shared[id], shared[id + s]);\n"};
        kernels.put(columnMaxs[0], getFunction(columnMaxs, reductionColumnFloats));

        String[] columnMins = {"columnMinsFloats",
                "\t\tshared[sIndex] = min(shared[sIndex], inputData[gid1 * rows + gid0]);\n",
                "\t\t\tshared[id] = min(shared[id], shared[id + s]);\n"};
        kernels.put(columnMins[0], getFunction(columnMins, reductionColumnFloats));

        String[] rowSums = {"rowSumsFloats",
                "\t\tshared[sIndex] += inputData[gid1 * rows + gid0];\n",
                "\t\t\tshared[tid1 * blockDim.x + tid0] += shared[(tid1 + s) * blockDim.x + tid0];\n"
        };
        kernels.put(rowSums[0], getFunction(rowSums, reductionRowFloats));

        String[] rowProducts = {"rowProductsFloats",
                "\t\tshared[sIndex] *= inputData[gid1 * rows + gid0];\n",
                "\t\t\tshared[tid1 * blockDim.x + tid0] *= shared[(tid1 + s) * blockDim.x + tid0];\n"
        };
        kernels.put(rowProducts[0], getFunction(rowProducts, reductionRowFloats));

        String[] rowMaxs = {"rowMaxsFloats",
                "\t\tshared[sIndex] = max(shared[sIndex], inputData[gid1 * rows + gid0]);\n",
                "\t\t\tunsigned int sharedIndex = tid1 * blockDim.x + tid0;\n" +
                        "\t\t\tshared[sharedIndex] = max(shared[sharedIndex], shared[(tid1 + s) * blockDim.x + tid0]);\n"
        };
        kernels.put(rowMaxs[0], getFunction(rowMaxs, reductionRowFloats));

        String[] rowMins = {"columnMinsFloats",
                "\t\tshared[sIndex] = min(shared[sIndex], inputData[gid1 * rows + gid0]);\n",
                "\t\t\tunsigned int sharedIndex = tid1 * blockDim.x + tid0;\n" +
                        "\t\t\tshared[sharedIndex] = min(shared[sharedIndex], shared[(tid1 + s) * blockDim.x + tid0]);\n"
        };
        kernels.put(rowMins[0], getFunction(rowMins, reductionRowFloats));
    }

    private static String getFunction(String[] function, String[] reduction) {
        StringBuilder builder = new StringBuilder();
        builder.append(reduction[0]);
        builder.append(function[0]);
        builder.append(reduction[1]);
        builder.append(function[1]);
        builder.append(reduction[2]);
        builder.append(function[2]);
        builder.append(reduction[3]);
        return builder.toString();
    }
}
