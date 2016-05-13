package org.nblas.cl;

import java.util.Random;

import javax.swing.plaf.FontUIResource;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;
import org.junit.Assert;
import org.nblas.Context;
import org.nblas.FloatMatrix;
import org.nblas.FloatMatrixTest;
import org.nblas.cl.model.CLArray;
import org.nblas.cl.model.CLScalar;
import org.nblas.cl.model.CLStorage;
import org.nblas.generic.Subprogram;
import org.nblas.impl.FloatMatrixDefault;

public class CLFloatMatrixSandbox extends FloatMatrixTest {

	protected static CLCore CORE;
	
	public static void main(String[] args) throws Exception {
		CLFloatMatrixSandbox testSuit = new CLFloatMatrixSandbox();
		testSuit.context = Context.createOpenCLSinglePrecisionContext();
		CORE = CLCore.getCore(testSuit.context.getDeviceId());
		
		testSuit.setUp();
		testSuit.sgemmBlockSize();		
	}
	
	public void functionTest() {
		float[] vals = new float[40*8];
		for (int i = 0; i < vals.length; i++) 
			vals[i] = i;
		CLFloatMatrix matrix = (CLFloatMatrix) FloatMatrix.create(40, 8, vals, context);
		FloatMatrix result = matrix.repmat(2, 3);
		System.out.println(matrix.toString2D());	
		System.out.println(result.toString2D());
		matrix.release();	
	}
	
	public void sgemmBlockSize() {
		int threadCount = CORE.getThreadCount();
    	cl_kernel kernel = CLPredefined.getSubprogram("sgemm_nn").getKernel();

		// erstelle eine 32x32 einser Matrix
		CLFloatMatrix matrix1 = (CLFloatMatrix) FloatMatrix.ones(1024, 1024, context).muli(3);
		CLFloatMatrix matrix2 = (CLFloatMatrix) FloatMatrix.ones(1024, 1024, context).muli(3);
		CLFloatMatrix matrix3 = (CLFloatMatrix) FloatMatrix.ones(1024, 1024, context).muli(3);

		
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
    
        CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
        CL.clSetKernelArg(kernel, 1, matrix2.getSizeof(), matrix2.getPointer());
        CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
        CL.clSetKernelArg(kernel, 3, Sizeof.cl_float*threadCount, CLArray.ofFloat(threadCount).getPointer());
        CL.clSetKernelArg(kernel, 4, Sizeof.cl_float*threadCount, CLArray.ofFloat(threadCount).getPointer());
        CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(matrix1.clRows).getPointer());
        CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, CLScalar.of(matrix2.clColumns).getPointer());
        CL.clSetKernelArg(kernel, 7, Sizeof.cl_uint, CLScalar.of(matrix1.clColumns).getPointer());
        
//        CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, 32, 32);	// 13ms
//        CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, 16, 16); // 15ms
//        CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, 8, 8); 	// 53ms
        
        CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, 64, 16);
        
//        CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, 32, 8);
//        CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, 8, 32);
//        CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, 256, 4);
//        CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, 16, 4);
        
		CORE.waitOnComplete();
		System.out.println("Matrix Multi Time: "+(System.currentTimeMillis()-start)+"ms");        
	}
	
	/**
	 * 
	 */
	public void matrixBlockSize() {
		
		String sourceCode = "" +
			"__kernel void times2(__global float* matrix, const uint clRows, const uint clColumns) \n" +
	    	"{\n" + 
	    	"    const uint column = get_global_id(0);\n" + // column
	    	"    const uint row = get_global_id(1);\n" + // row
	    	"\n" +
	        "	 matrix[row + column * clRows] = sqrt(matrix[row + column * clRows]);\n" +
	    	"}\n";

		
		Subprogram<cl_kernel> subprogram = new Subprogram<>("times2", sourceCode, true);
		CORE.loadFromGeneratedSubprogram(subprogram);
		cl_kernel kernel = subprogram.getKernel();
			
		// erstelle eine 32x32 einser Matrix
//		CLFloatMatrix matrix1 = (CLFloatMatrix) FloatMatrix.ones(8, 128 * 10000, context).muli(3);
//		CLFloatMatrix matrix2 = (CLFloatMatrix) FloatMatrix.ones(8, 128 * 10000, context).muli(3);
		CLFloatMatrix matrix1 = (CLFloatMatrix) FloatMatrix.ones(16, 128 * 100000, context).muli(3);
		CLFloatMatrix matrix2 = (CLFloatMatrix) FloatMatrix.ones(16, 128 * 100000, context).muli(3);
//		CLFloatMatrix matrix1 = (CLFloatMatrix) FloatMatrix.ones(32, 128 * 10000, context).muli(3);
//		CLFloatMatrix matrix2 = (CLFloatMatrix) FloatMatrix.ones(32, 128 * 10000, context).muli(3);
		
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
    
        CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_uint, CLScalar.of(matrix1.clRows).getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, CLScalar.of(matrix1.clColumns).getPointer());
//        CORE.enqueue2DRangeKernelTest(kernel, 8, matrix1.clColumns, 8, 128);
//        CORE.enqueue2DRangeKernelTest(kernel, 8, matrix1.clColumns, 8, 64);
//        CORE.enqueue2DRangeKernelTest(kernel, 8, matrix1.clColumns, 8, 32);
//        CORE.enqueue2DRangeKernelTest(kernel, 8, matrix1.clColumns, 8, 16);
//        CORE.enqueue2DRangeKernelTest(kernel, 8, matrix1.clColumns, 8, 8);
//        CORE.enqueue2DRangeKernelTest(kernel, 8, matrix1.clColumns, 8, 4);
//        CORE.enqueue2DRangeKernelTest(kernel, 8, matrix1.clColumns, 8, 2);
//        CORE.enqueue2DRangeKernelTest(kernel, 8, matrix1.clColumns, 8, 1);
        
//        CORE.enqueue2DRangeKernelTest(kernel, 16, matrix1.clColumns, 16, 64);
//        CORE.enqueue2DRangeKernelTest(kernel, 16, matrix1.clColumns, 16, 32);
//        CORE.enqueue2DRangeKernelTest(kernel, 16, matrix1.clColumns, 16, 16);
//        CORE.enqueue2DRangeKernelTest(kernel, 16, matrix1.clColumns, 16, 8);
        CORE.enqueue2DRangeKernelTest(kernel, 16, matrix1.clColumns, 16, 4);
//        CORE.enqueue2DRangeKernelTest(kernel, 16, matrix1.clColumns, 16, 2);
//        CORE.enqueue2DRangeKernelTest(kernel, 16, matrix1.clColumns, 16, 1);
        
//        CORE.enqueue2DRangeKernelTest(kernel, 32, matrix1.clColumns, 32, 32);
//        CORE.enqueue2DRangeKernelTest(kernel, 32, matrix1.clColumns, 32, 16);
//        CORE.enqueue2DRangeKernelTest(kernel, 32, matrix1.clColumns, 32, 8);
//        CORE.enqueue2DRangeKernelTest(kernel, 32, matrix1.clColumns, 32, 4);
//        CORE.enqueue2DRangeKernelTest(kernel, 32, matrix1.clColumns, 32, 2);
//        CORE.enqueue2DRangeKernelTest(kernel, 32, matrix1.clColumns, 32, 1);
        
		CORE.waitOnComplete();
		System.out.println("Matrix1 Time: "+(System.currentTimeMillis()-start)+"ms");
		
		// Teste die zweite Variante
		CORE.waitOnComplete();
		start= System.currentTimeMillis();		
		
        CL.clSetKernelArg(kernel, 0, matrix2.getSizeof(), matrix2.getPointer());
        CL.clSetKernelArg(kernel, 1, Sizeof.cl_uint, CLScalar.of(matrix2.clRows).getPointer());
        CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, CLScalar.of(matrix2.clColumns).getPointer());
        CORE.enqueue2DRangeKernel(kernel, matrix2.clRows, matrix2.clColumns, 0, 0);
		
		CORE.waitOnComplete();
		System.out.println("Matrix2 Time: "+(System.currentTimeMillis()-start)+"ms");

		// vergleichen und sauber machen
        float[] result = matrix1.toArray();
        float testVal = result[0];
        for (int i = 0; i < result.length; i++)
			if(result[i] != testVal)
				System.out.println("fehler");  
        
        assertAndFree(matrix1, matrix2);        
        System.out.println("Finished");
	}
	
	/**
	 * Ein Kernel kann mehrmals hintereinander ausgeführt werden.
	 */
	public void multiKernelRuns() {
		
		// multipliziere die Werte in der Matrix *2
		String sourceCode = "" +
			"__kernel void times2(__global float* matrix) \n" +
	    	"{\n" + 
	    	"    const uint gid0 = get_global_id(0);\n" +
	    	"    const uint gid1 = get_global_id(1);\n" + 
	    	"	 const uint globalSize0 = get_local_size(0);\n" + 
	    	"\n" +
	        "	 matrix[gid0 + gid1 * globalSize0] *= 2;\n" +
	    	"}\n";

		
		Subprogram<cl_kernel> subprogram = new Subprogram<>("times2", sourceCode, true);
		CORE.loadFromGeneratedSubprogram(subprogram);
			
		// erstelle eine 32x32 einser Matrix
		CLFloatMatrix matrix = (CLFloatMatrix) FloatMatrix.ones(32, 32, context);
		
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();
		CORE.execute(subprogram, matrix.clRows, matrix.clColumns, matrix);
		CORE.waitOnComplete();
		System.out.println("time: "+(System.currentTimeMillis()-start)+"ms");
		
		System.out.println(matrix.toString2D());	
		matrix.release();			
	}

	public void groupIdTest() {
		
		// 1D http://web.engr.oregonstate.edu/~mjb/cs575/Handouts/opencl.reduction.1pp.pdf
		String sourceCode = "" +
			"__kernel void groupIdTest(__global float* matrix, local float *prods, __global float *dC) \n" +
	    	"{\n" + 
	    	"    const uint tid0 = get_local_id(0);\n" + // 0 bis 0
	    	"    const uint gid0 = get_global_id(0);\n" + // 0 bis 0
	    	"    const uint gid1 = get_global_id(1);\n" + // 0 bis 31
	    	"    const uint localSize0 = get_local_size(0);\n" + // 1
	    	"    const uint workGroupId0 = get_group_id(0);\n" + // work group id
	    	"\n" +
	        "	 prods[tid0] = matrix[gid0];\n" +
	        "\n" +
	        "	// all threads execute this code simultaneously:\n" +
	        "	for(int offset = 1; offset < localSize0; offset *= 2)\n" +
	        "	{\n" +
	        "		int mask = 2*offset - 1;\n" +
	        "		barrier(CLK_LOCAL_MEM_FENCE); // wait for completion\n" +
	        "		if((tid0 & mask) == 0)\n" +
	        "		{\n" +
	        "			prods[tid0] += prods[tid0 + offset];\n" +
	        "		}\n" +
	        "	}\n" +
	        "\n" +
	        "	barrier(CLK_LOCAL_MEM_FENCE);\n" +
	        "	if(tid0 == 0)\n" +
	        "		dC[workGroupId0] = prods[0];\n" +
	    	"}\n";

		
		Subprogram<cl_kernel> subprogram = new Subprogram<>("groupIdTest", sourceCode, true);
		CORE.loadFromGeneratedSubprogram(subprogram);
			
		int size = 1000000;
		org.jblas.FloatMatrix cpuMat = new org.jblas.FloatMatrix(1, size);
		for (int i = 0; i < cpuMat.columns; i++)
			cpuMat.data[i] = (((float)size)/(i+1));
		long start= System.currentTimeMillis();
		float cpuSum = cpuMat.sum();
		System.out.println("cpu sum "+cpuSum+" took "+(System.currentTimeMillis()-start));
		
		// erstelle die Matrix
		CLFloatMatrix matrix = (CLFloatMatrix) FloatMatrix.create(cpuMat.toArray2(), context);
		CLFloatMatrix matrixOut = (CLFloatMatrix) FloatMatrix.zeros(1, (cpuMat.columns/32)+1, context);
		
		// führe den Kernel aus
		CORE.waitOnComplete();
		start= System.currentTimeMillis();
//		CLFloatMatrix.testsum(matrix);
		CORE.execute(subprogram, matrix.clRows, matrix.clColumns, matrix, CLArray.ofFloat(1024), matrixOut);
		CORE.waitOnComplete();
		System.out.println("time: "+(System.currentTimeMillis()-start)+"ms");
		org.jblas.FloatMatrix gpuSum = new org.jblas.FloatMatrix(matrixOut.toArray2());
		System.out.println("gpu sum "+gpuSum.sum());
		
		System.out.println(matrix.toString2D());	
		System.out.println(matrixOut.toString2D());	
		matrixOut.release();
		matrix.release();			
	}
	
	// TODO 3x4 matrix mmul 1x3 = 1x4
	// performance
	public void smallMmulTest() {
		
		String sourceCode = "" +
			"__kernel void smallMmulTest(const __global float* a, const __global float* b, __global float* c, \n" +
            "           __local float* aSub, __local float* bSub, \n" +
            "           const int M, const int N, const int K) \n" +
    		"{\n" + 
    		"    const int tid0 = get_local_id(0);\n" + // 0 bis 0
    		"    const int tid1 = get_local_id(1);\n" + // 0 bis 31
    		"    const int gid0 = get_global_id(0);\n" + // 0 bis 0
    		"    const int gid1 = get_global_id(1);\n" + // 0 bis 31
    		"    int localSize0 = get_local_size(0);\n" + // 1
    		"    int localSize1 = get_local_size(1);\n" + // 32
    		"\n" +
    	    "	float result = 0.0f;\n" +
    	    "	int numTiles = K / localSize0;\n" + // 32
    	    "	int index = tid1 * localSize0 + tid0;\n" + // 0 bis 31
    	    "	for (int t =0; t < numTiles; t++) {\n" + // 0 bis 31
    	    "	    int tiled0 = localSize0 * t + tid0;\n" + // 0 bis 31
    	    "	    int tiled1 = localSize1 * t + tid1;\n" + // 32*0...31 + 0...31
    	    "	    aSub[index] = a[tiled1 * M + gid0];\n" +
    	    "	    bSub[index] = b[gid1 * K + tiled0];\n" +
    	    "	    barrier(CLK_LOCAL_MEM_FENCE);\n" +
    	    "	    for (int k = 0; k < localSize0; k++) {\n" + // wird durchlaufen auch wenn aSub und bSub noch nicht gefüllt sind
    	    "	        result += aSub[k * localSize0 + tid0] * bSub[tid1 * localSize0 + k];\n" +
    	    "	    }\n" +
    	    "	    barrier(CLK_LOCAL_MEM_FENCE);\n" +
    	    "	}\n" +
    	    "	c[gid1 * M + gid0] = result;\n" +
    		"}\n";
		
		Subprogram<cl_kernel> subprogram = new Subprogram<>("smallMmulTest", sourceCode, true);
		CORE.loadFromGeneratedSubprogram(subprogram);
		
		float[][] input = new float[][] { {1,0,0},{1,0,1},{1,1,0},{1,1,1} };
		float[][] weights = new float[][] { {1.7f}, {0.14f}, {0.71f} };
		
		CLFloatMatrix inputMat = (CLFloatMatrix) FloatMatrix.create(input, context);
		CLFloatMatrix weightMat = (CLFloatMatrix) FloatMatrix.create(weights, context);
		CLFloatMatrix outputMat = (CLFloatMatrix) FloatMatrix.zeros(1, 4, context);
		
		inputMat.mmulCustom(subprogram, inputMat, weightMat, outputMat);
		inputMat.mmul(inputMat, weightMat, outputMat);
		
		System.out.println(outputMat.toString2D());
		
		inputMat.release();
		weightMat.release();
		outputMat.release();
	}
	
	
	public void gTest() {
		String sourceCode = "" +
			"__kernel void gTest(__global const float* source, __global float* destination, const uint dstRows, const uint dstColumns, const uint rowOffset, const uint columnOffset, const uint srcStride)\n" +
    		"{\n" + 
    		"    uint dstX = get_global_id(0);\n" + 
    		"    uint dstY = get_global_id(1);\n" +
    		"	 if(dstX >= dstColumns || dstY >= dstRows) return;\n" +
    		"\n" +
    		"    uint dstIndex = dstX * get_global_size(1) + dstY;" + 
    		"    uint srcIndex = (dstX + columnOffset) * srcStride + (dstY + rowOffset);" +
    		"    destination[dstIndex] = source[srcIndex];\n" +
    		"}\n";
		
		Subprogram<cl_kernel> subprogram = new Subprogram<>("gTest", sourceCode, true);
		CORE.loadFromGeneratedSubprogram(subprogram);
		
		CLFloatMatrix clMat = (CLFloatMatrix) FloatMatrix.zeros(1, matA_GPU.getColumns(), context);    	
		clMat.getCustom(subprogram, matA_GPU, clMat, 10, 0);
//		clMat.getSubMatrix(matA_GPU, clMat, 60, 1);
		
		System.out.println(matA_GPU.toString2D());
		System.out.println(clMat.toString2D());
		
		clMat.release();
	}
	
	public void customKernelTest() {
						
		String sourceCode = ""
				+ "__kernel void customTest(__global float* result, const uint columns, const uint rows, __global const float* matrix, __global const float* rowVec)"
				+ "{"
				+ "    uint y = get_global_id(0);"
				+ "    uint x = get_global_id(1);"
				+ "    if (y >= rows || x >= columns) return;"
				+ "    uint id = x * get_global_size(0) + y;"
				+ "    result[id] = matrix[id] + rowVec[y];"
				+ "}";
		
		Subprogram<cl_kernel> subprogram = new Subprogram<>("customTest", sourceCode, true);
		CORE.loadFromGeneratedSubprogram(subprogram);
		
		CLFloatMatrix clMat = (CLFloatMatrix) FloatMatrix.zeros(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		clMat.runMatrixRowVectorElementWiseOperation(subprogram, (CLFloatMatrix) rowVector_GPU, clMat);
		
		System.out.println(rowVector_GPU.toString2D());
		System.out.println(clMat.toString2D());
		
		clMat.release();
	}
	
	
	public void sumWithDifferentSizesTest() {
		
		for (int size = 1; size < 1000; size+=10) {
			
			Random rnd = new Random(seed);
			float[] matCFloatArray = new float[size * size];
			for (int i = 0; i < matCFloatArray.length; i++)
				matCFloatArray[i] = rnd.nextFloat();

			org.jblas.FloatMatrix matC_CPU = new org.jblas.FloatMatrix(size, size, matCFloatArray);
			FloatMatrix matC_GPU = FloatMatrix.create(size, size, matCFloatArray, context);

			// Berechnung auf der CPU
			float sum_CPU = matC_CPU.sum();

			// Berechnung auf der GPU
			float sum_GPU = matC_GPU.sum();

			// erlaube etwas Ungenauigkeit			
			Assert.assertEquals(sum_CPU, sum_GPU, Math.pow(((float)size/1000)*4,2));
			System.out.println("Compare: "+sum_CPU+" vs "+sum_GPU);


			matC_GPU.release();
		}
	}
	
	public void sumAfterwardsTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.ones(matA_CPU.getRows(), matA_CPU.getColumns()).muli(2);
		for (int i = 0; i < 10; i++)
			matC_CPU.muli(2);
		float sum_CPU = matC_CPU.sum();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU1 = FloatMatrixDefault.dirtyAllocation(1024, 1024, context);
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(4, 4, context);
//		matC_GPU.addi(1);
//		matC_GPU.addi(Float.NaN);
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns(), context);
//		FloatMatrix matC_GPU = FloatMatrix.ones(matA_GPU.getRows(), matA_GPU.getColumns(), context).muli(2);
		
		System.out.println(matC_GPU.toString2D());
		
//		for (int i = 0; i < 10; i++) {
//			matC_GPU.muli(2);
//			CLCore.getCore().waitOnComplete();
//		}
		float sum_GPU = matC_GPU.sum();
		
		System.out.println("Compare: "+sum_CPU+" vs "+sum_GPU);

		System.out.println(matA_GPU.getRows() * matA_GPU.getColumns() * Math.pow(2, 11));
		Assert.assertEquals(sum_CPU, sum_GPU, 0.1f);
		matC_GPU1.release();	
		matC_GPU.release();	
	}
	
	
}
