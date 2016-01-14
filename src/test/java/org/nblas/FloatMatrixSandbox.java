package org.nblas;

import java.util.Random;

import org.junit.Assert;

public class FloatMatrixSandbox extends FloatMatrixTest {

	public static void main(String[] args) throws Exception {
		FloatMatrixSandbox testSuit = new FloatMatrixSandbox();
		testSuit.setUp();
		testSuit.sumWithDifferentSizesTest();
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

			matC_GPU.free();
		}
	}
	
	public void sumAfterwardsTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.ones(matA_CPU.getRows(), matA_CPU.getColumns()).muli(2);
		for (int i = 0; i < 10; i++)
			matC_CPU.muli(2);
		float sum_CPU = matC_CPU.sum();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrix.ones(matA_GPU.getRows(), matA_GPU.getColumns(), context).muli(2);
		for (int i = 0; i < 10; i++)
			matC_GPU.muli(2);
		float sum_GPU = matC_GPU.sum();
		
		Assert.assertEquals(sum_CPU, sum_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	
	
}
