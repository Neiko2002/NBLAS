package org.nblas;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

import org.jblas.ranges.IntervalRange;
import org.jblas.ranges.RangeUtils;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.rules.Stopwatch;
import org.nblas.cl.CLFloatMatrix;

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
		testSuit.gtScalarTest();
	}
	
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
		matA_GPU = new FloatMatrix(matARows, matAColumns, matAFloatArray);
		
		matB_CPU = new org.jblas.FloatMatrix(matBRows, matBColumns, matBFloatArray);
		matB_GPU = new FloatMatrix(matBRows, matBColumns, matBFloatArray);
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
		
		for (int size = 1; size < 1000; size++) {
			
			Random rnd = new Random(seed);
			float[] matCFloatArray = new float[size * size];
			for (int i = 0; i < matCFloatArray.length; i++)
				matCFloatArray[i] = rnd.nextFloat();

			org.jblas.FloatMatrix matC_CPU = new org.jblas.FloatMatrix(size, size, matCFloatArray);
			FloatMatrix matC_GPU = new FloatMatrix(size, size, matCFloatArray);

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
		FloatMatrix matC_GPU = FloatMatrix.ones(matA_GPU.getRows(), matA_GPU.getColumns()).muli(2);
		for (int i = 0; i < 10; i++)
			matC_GPU.muli(2);
		float sum_GPU = matC_GPU.sum();
		
		Assert.assertEquals(sum_CPU, sum_GPU, 0.1f);
		
		matC_GPU.free();
	}
}
