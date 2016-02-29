package org.nblas.cl;

import java.util.Random;

import org.jocl.cl_kernel;
import org.junit.Assert;
import org.junit.Test;
import org.nblas.Context;
import org.nblas.FloatMatrix;
import org.nblas.FloatMatrixDefault;
import org.nblas.FloatMatrixTest;
import org.nblas.generic.Subprogram;

public class CLFloatMatrixSandbox extends FloatMatrixTest {

	
	public static void main(String[] args) throws Exception {
		CLFloatMatrixSandbox testSuit = new CLFloatMatrixSandbox();
		testSuit.context = Context.createOpenCLSinglePrecisionContext();
		testSuit.setUp();
		testSuit.gTest();
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
		CLCore CORE = CLCore.getCore();
		CORE.loadFromGeneratedSubprogram(subprogram);
		
		CLFloatMatrix clMat = (CLFloatMatrix) FloatMatrix.zeros(1, matA_GPU.getColumns(), context);
		clMat.getCustom(subprogram.getKernel(), matA_GPU, clMat, 60, 0);
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
		CLCore CORE = CLCore.getCore();
		CORE.loadFromGeneratedSubprogram(subprogram);
		
		CLFloatMatrix clMat = (CLFloatMatrix) FloatMatrix.zeros(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		CLFloatMatrix.runMatrixRowVectorElementWiseOperation(subprogram, clMat, (CLFloatMatrix) rowVector_GPU, clMat);
		
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
