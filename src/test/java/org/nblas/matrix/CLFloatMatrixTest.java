package org.nblas.matrix;

import java.util.Random;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class CLFloatMatrixTest {

	protected static final int seed = 7;
	
	protected static final int matARows = 16;
	protected static final int matAColumns = 16;
	
	protected static final int matBRows = 16;
	protected static final int matBColumns = 16;
	
	public static void main(String[] args) throws Exception {
		CLFloatMatrixTest testSuit = new CLFloatMatrixTest();
		testSuit.setUp();
		testSuit.mmulTest();
	}
	
	protected org.jblas.FloatMatrix matA_CPU;
	protected org.jblas.FloatMatrix matB_CPU;
	
	protected CLFloatMatrix matA_GPU;
	protected CLFloatMatrix matB_GPU;

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
		matA_GPU = new CLFloatMatrix(matARows, matAColumns, matAFloatArray);
		
		matB_CPU = new org.jblas.FloatMatrix(matBRows, matBColumns, matBFloatArray);
		matB_GPU = new CLFloatMatrix(matBRows, matBColumns, matBFloatArray);
	}

	@After
	public void release() throws Exception {
		matA_GPU.free();
		matB_GPU.free();
	}
	
	@Test
	public void mmulTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mmul(matB_CPU);
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matB_GPU.getColumns());
		CLFloatMatrix.mmul(matA_GPU, matB_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
	}

}
