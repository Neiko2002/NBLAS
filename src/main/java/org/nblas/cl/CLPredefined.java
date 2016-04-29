package org.nblas.cl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.jocl.cl_kernel;
import org.nblas.generic.Subprogram;

/**
 * Created by Moritz on 7/7/2015.
 * 
 * TODO sind nur gültig für float Matrizen
 * 
 * TODO sollte ausgelagert werden in einzelne cl Dateien
 * 
 * https://github.com/sschaetz/nvidia-opencl-examples/blob/master/OpenCL/src/oclTranspose/transpose.cl
 * 
 */
class CLPredefined {
		
	private static final String repmat = "__kernel void repmat(const __global float* source, __global float* destination, const uint srcRows, const uint srcColumns, const uint srcStride)\n" +
		"{\n" +
		"    const uint dstX = get_global_id(0);\n" +
		"    const uint dstY = get_global_id(1);\n" +
		"    const uint idx = dstX * get_global_size(1) + dstY;\n" +
		"    const uint sidx = (dstY % srcRows) + (dstX % srcColumns) * srcStride;\n" +
		"    destination[idx] = source[sidx];\n" +
		"}\n";
    
    private static final String setSubMatrix = "__kernel void setSubMatrix(const __global float* source, __global float* destination, const uint srcRows, const uint srcColumns, const uint rowOffset, const uint columnOffset, const uint dstStride)\n" +
    		"{\n" + 
    		"    const uint srcX = get_global_id(0);\n" +
    		"    const uint srcY = get_global_id(1);\n" + 
    		"	 if(srcX >= srcColumns || srcY >= srcRows) return;\n" +
    		"\n" +
    		"    const uint srcIndex = srcX * get_global_size(1) + srcY;" + 
    		"    const uint dstIndex = (srcX + columnOffset) * dstStride + (srcY + rowOffset);" +
    		"    destination[dstIndex] = source[srcIndex];\n" +
    		"}\n";
    
    private static final String getSubMatrix = "__kernel void getSubMatrix(const __global float* source, __global float* destination, const uint dstRows, const uint dstColumns, const uint rowOffset, const uint columnOffset, const uint srcStride)\n" +
    		"{\n" + 
    		"    const uint dstX = get_global_id(0);\n" + 
    		"    const uint dstY = get_global_id(1);\n" +
    		"	 if(dstX >= dstColumns || dstY >= dstRows) return;\n" +
    		"\n" +
    		"    const uint dstIndex = dstX * get_global_size(1) + dstY;" + 
    		"    const uint srcIndex = (dstX + columnOffset) * srcStride + (dstY + rowOffset);" +
    		"    destination[dstIndex] = source[srcIndex];\n" +
    		"}\n";
	
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

    private static final String[] sumFloats = {
    	"__kernel void ",
	    "(__global float* buffer, __global float* result,  __local float* shared, __const int length)\n" +
	    "{\n" +
	    "	int global_index = get_global_id(0);\n" +
	    "	int global_size = get_global_size(0);\n" +
	    "	float accumulator = 0.0f;\n" +
	    "\n" +
	    "	// Loop sequentially over chunks of input vector\n" +
	    "	while (global_index < length) {\n" +
	    "		accumulator += buffer[global_index];\n" +
	    "		global_index += global_size;\n" +
	    "	}\n" +
	    "\n" +
	    "	// Perform parallel reduction\n" +
	    "	int local_index = get_local_id(0);\n" +
	    "	shared[local_index] = accumulator;\n" +
	    "	barrier(CLK_LOCAL_MEM_FENCE);\n" +
	    "	for(int offset = get_local_size(0) >> 1; offset > 0; offset >>= 1) {\n" +
	    "		if (local_index < offset) {\n" +
	    "			shared[local_index] += shared[local_index + offset];\n" +
	    "		}\n" +
	    "		barrier(CLK_LOCAL_MEM_FENCE);\n" +
	    "	}\n" +
	    "\n" +
	    "	if (local_index == 0) {\n" +
	    "		result[get_group_id(0)] = shared[0];\n" +
	    "	}\n" +
	    "}"
    };
    
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

    private static final String copyColumnMajor = "__kernel void copyColumnMajor(__global float* input, __global float* output, uint n)\n" +
            "{\n" +
            "   int gid = get_global_id(0);\n" +
            "   if(gid >= n) return;\n" +
            "   output[gid] = input[gid];\n" +
            "}\n";
    
    private static final String copyRowMajor = "__kernel void copyRowMajor(__global float* input, __global float* output, uint n, uint clRows)\n" +
            "{\n" +
            "   int gid = get_global_id(0);\n" +
            "   if(gid >= n) return;\n" +
            "   output[gid*clRows] = input[gid];\n" +
            "}\n";

    private static final String transposeInPlace = "__kernel void transposeInPlace(const __global float* input,\n" +
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
    
    private static final String transpose = "__kernel void transpose(const __global float* input, __global float* output, int rows, int columns) {\n" +
            "\n" +
            "    const int column = get_global_id(0);\n" +
            "    const int row = get_global_id(1);\n" +
            "    output[row * get_global_size(0) + column] = input[column * get_global_size(1) + row];\n" + 
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

    private static final List<Subprogram<cl_kernel>> subprograms;
    private static final HashMap<String, Subprogram<cl_kernel>> nameToSubprogramMap;

    static {
        subprograms = new ArrayList<>();
        nameToSubprogramMap = new HashMap<>();
        addSubprogram(new Subprogram<cl_kernel>("repmat", repmat, false));
        addSubprogram(new Subprogram<cl_kernel>("getSubMatrix", getSubMatrix, false));
        addSubprogram(new Subprogram<cl_kernel>("setSubMatrix", setSubMatrix, false));
        addSubprogram(new Subprogram<cl_kernel>("auniform", uniform, false));
        addSubprogram(new Subprogram<cl_kernel>("boxmuller", boxmuller, false));
        addSubprogram(new Subprogram<cl_kernel>("copyColumnMajor", copyColumnMajor, false));
        addSubprogram(new Subprogram<cl_kernel>("copyRowMajor", copyRowMajor, false));
        addSubprogram(new Subprogram<cl_kernel>("transposeInPlace", transposeInPlace, false));
        addSubprogram(new Subprogram<cl_kernel>("transpose", transpose, false));
        addSubprogram(new Subprogram<cl_kernel>("setZero", setZero, false));
        addSubprogram(new Subprogram<cl_kernel>("sgemm_nn", sgemm_nn, false));
        addSubprogram(new Subprogram<cl_kernel>("sgemm_nt", sgemm_nt, false));
        addSubprogram(new Subprogram<cl_kernel>("sgemm_tn", sgemm_tn, false));
        String[] sum = {"sumFloats",
                "\t\tshared[sIndex] += inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\tshared[sIndex] += shared[sIndex + s];\n"};
        addSubprogram(new Subprogram<cl_kernel>(sum[0], buildReductionKernel(sum, reductionFloats), false));

        String[] product = {"productFloats",
                "\t\tshared[sIndex] *= inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\tshared[sIndex] *= shared[sIndex + s];\n"};
        addSubprogram(new Subprogram<cl_kernel>(product[0], buildReductionKernel(product, reductionFloats), false));

        String[] max = {"maxFloats",
                "\t\tshared[sIndex] = max(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tshared[sIndex] = max(shared[sIndex], shared[sIndex + s]);\n"};
        addSubprogram(new Subprogram<cl_kernel>(max[0], buildReductionKernel(max, reductionFloats), false));

        String[] min = {"minFloats",
                "\t\tshared[sIndex] = min(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tshared[sIndex] = min(shared[sIndex], shared[sIndex + s]);\n"};
        addSubprogram(new Subprogram<cl_kernel>(min[0], buildReductionKernel(min, reductionFloats), false));

        String[] columnSums = {"columnSumsFloats",
                "\t\tshared[sIndex] += inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\t\tshared[id] += shared[id + s];\n"};
        addSubprogram(new Subprogram<cl_kernel>(columnSums[0], buildReductionKernel(columnSums, reductionColumnFloats), false));

        String[] columnProducts = {"columnProductsFloats",
                "\t\tshared[sIndex] *= inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\t\tshared[id] *= shared[id + s];\n"};
        addSubprogram(new Subprogram<cl_kernel>(columnProducts[0], buildReductionKernel(columnProducts, reductionColumnFloats), false));

        String[] columnMaxs = {"columnMaxsFloats",
                "\t\tshared[sIndex] = max(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tshared[id] = max(shared[id], shared[id + s]);\n"};
        addSubprogram(new Subprogram<cl_kernel>(columnMaxs[0], buildReductionKernel(columnMaxs, reductionColumnFloats), false));

        String[] columnMins = {"columnMinsFloats",
                "\t\tshared[sIndex] = min(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tshared[id] = min(shared[id], shared[id + s]);\n"};
        addSubprogram(new Subprogram<cl_kernel>(columnMins[0], buildReductionKernel(columnMins, reductionColumnFloats), false));

        String[] rowSums = {"rowSumsFloats",
                "\t\tshared[sIndex] += inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\t\tshared[tid1 * get_local_size(0) + tid0] += shared[(tid1 + s) * get_local_size(0) + tid0];\n"
        };
        addSubprogram(new Subprogram<cl_kernel>(rowSums[0], buildReductionKernel(rowSums, reductionRowFloats), false));

        String[] rowProducts = {"rowProductsFloats",
                "\t\tshared[sIndex] *= inputData[gid1 * get_global_size(0) + gid0];\n",
                "\t\t\tshared[tid1 * get_local_size(0) + tid0] *= shared[(tid1 + s) * get_local_size(0) + tid0];\n"
        };
        addSubprogram( new Subprogram<cl_kernel>(rowProducts[0], buildReductionKernel(rowProducts, reductionRowFloats), false));

        String[] rowMaxs = {"rowMaxsFloats",
                "\t\tshared[sIndex] = max(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tunsigned int sharedIndex = tid1 * get_local_size(0) + tid0;\n" +
                        "\t\t\tshared[sharedIndex] = max(shared[sharedIndex], shared[(tid1 + s) * get_local_size(0) + tid0]);\n"
        };
        addSubprogram(new Subprogram<cl_kernel>(rowMaxs[0], buildReductionKernel(rowMaxs, reductionRowFloats), false));

        String[] rowMins = {"rowMinsFloats",
                "\t\tshared[sIndex] = min(shared[sIndex], inputData[gid1 * get_global_size(0) + gid0]);\n",
                "\t\t\tunsigned int sharedIndex = tid1 * get_local_size(0) + tid0;\n" +
                        "\t\t\tshared[sharedIndex] = min(shared[sharedIndex], shared[(tid1 + s) * get_local_size(0) + tid0]);\n"
        };
        addSubprogram(new Subprogram<cl_kernel>(rowMins[0], buildReductionKernel(rowMins, reductionRowFloats), false));
        
        String[] sumValues = {"sumFloats1D"};
        addSubprogram(new Subprogram<cl_kernel>(sumValues[0], buildReductionKernel(sumValues, sumFloats), false));
    }
    
    private static void addSubprogram(Subprogram<cl_kernel> subprogram) {
        subprograms.add(subprogram);
        nameToSubprogramMap.put(subprogram.getProgramName(), subprogram);
    }
    
    public static List<Subprogram<cl_kernel>>getAllSubPrograms() {
    	return subprograms;
    }
    
    public static Subprogram<cl_kernel> getSubprogram(String name) {
    	return nameToSubprogramMap.get(name);
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
