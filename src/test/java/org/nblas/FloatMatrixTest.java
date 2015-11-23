package org.nblas;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

import org.jblas.MatrixFunctions;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class FloatMatrixTest {

	protected static final int seed = 7;
	protected static final int runs = 100_000;
	
	protected static final int matARows = 256;
	protected static final int matAColumns = 256;
	
	protected static final int matBRows = 256;
	protected static final int matBColumns = 256;
	
	public static void main(String[] args) throws Exception {
		FloatMatrixTest testSuit = new FloatMatrixTest();
		testSuit.setUp();
		testSuit.sumAfterwardsTest();
	}
	
	protected Context context = Context.createOpenCLSinglePrecisionContext();
	
	protected org.jblas.FloatMatrix matA_CPU;
	protected org.jblas.FloatMatrix matB_CPU;
	
	protected FloatMatrix matA_GPU;
	protected FloatMatrix matB_GPU;

	@Before
	public void setUp() throws Exception {
		Random rnd = new Random(seed);
		
		// Test-Daten anlegen
		float[] matAFloatArray = new float[matARows*matAColumns];
		float[] matBFloatArray = new float[matBRows*matBColumns];
		
		// Arrays mit Zufallszahlen füllen
		for (int i = 0; i < matAFloatArray.length; i++) 
			matAFloatArray[i] = rnd.nextFloat();

		for (int i = 0; i < matBFloatArray.length; i++) 
			matBFloatArray[i] = rnd.nextFloat();
		
		// die Daten auf die Grafikkarte kopieren
		matA_CPU = new org.jblas.FloatMatrix(matARows, matAColumns, matAFloatArray);
		matA_GPU = FloatMatrix.create(matARows, matAColumns, matAFloatArray, context);
		
		matB_CPU = new org.jblas.FloatMatrix(matBRows, matBColumns, matBFloatArray);
		matB_GPU = FloatMatrix.create(matBRows, matBColumns, matBFloatArray, context);
	}

	@After
	public void release() throws Exception {
		matA_GPU.free();
		matB_GPU.free();
	}

	@Test
	public void gtScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.gt(0);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.gt(0);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void memoryLeakTest() {
		
		Instant start = Instant.now();
		for (int i = 0; i < runs; i++) {			
			matA_GPU.addi(matB_GPU);
		}
		System.out.println("took "+Duration.between(start, Instant.now()));
	}
	
	@Test
	public void sumTest() {
		
		// Berechnung auf der CPU
		float sum_CPU = matA_CPU.sum();
		
		// Berechnung auf der GPU
		float sum_GPU = matA_GPU.sum();
		
		Assert.assertEquals(sum_CPU, sum_GPU, 1.0f);
	}
	
	@Test
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
	
	@Test
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
	
	@Test
	public void expTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = MatrixFunctions.exp(matA_CPU);

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.exp();
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void expiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = MatrixFunctions.exp(matA_CPU);

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().expi();
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void expMatrixTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = MatrixFunctions.exp(matA_CPU);

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrix.zeros(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.exp(matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void negateTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.neg();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.neg();
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void negateInPlaceTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.neg();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().negi();
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void negateMatrixTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.neg();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrix.zeros(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.neg(matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void sigmoidTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup();
		for (int i = 0; i < matA_CPU.data.length; i++)
			matC_CPU.data[i] = (float) (1. / ( 1. + Math.exp(-matA_CPU.data[i]) ));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.sigmoid();
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void sigmoidInPlaceTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup();
		for (int i = 0; i < matA_CPU.data.length; i++)
			matC_CPU.data[i] = (float) (1. / ( 1. + Math.exp(-matA_CPU.data[i]) ));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().sigmoidi();
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void sigmoidMatrixTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup();
		for (int i = 0; i < matA_CPU.data.length; i++)
			matC_CPU.data[i] = (float) (1. / ( 1. + Math.exp(-matA_CPU.data[i]) ));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrix.zeros(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.sigmoid(matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void duplicateTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
}
