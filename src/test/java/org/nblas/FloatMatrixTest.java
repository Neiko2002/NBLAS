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
	
	protected static final int matARows = 16;
	protected static final int matAColumns = 16;
	
	protected static final int matBRows = 16;
	protected static final int matBColumns = 16;
	
	public static void main(String[] args) throws Exception {
		FloatMatrixTest testSuit = new FloatMatrixTest();
		testSuit.setUp();
		testSuit.memoryLeakTest();
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
	public void memoryLeakTest() {
		
		Instant start = Instant.now();
		for (int i = 0; i < runs; i++) {			
			matA_GPU.addi(matB_GPU);
		}
		System.out.println("took "+Duration.between(start, Instant.now()));
	}
	

	
}
