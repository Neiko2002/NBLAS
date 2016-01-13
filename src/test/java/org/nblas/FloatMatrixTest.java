package org.nblas;

import java.util.Random;

import org.jblas.MatrixFunctions;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class FloatMatrixTest {

	protected static final int seed = 7;
	protected static final int runs = 100_000;
	protected static final int matrixSize = 256; 
	
	protected static final int matARows = matrixSize;
	protected static final int matAColumns = matrixSize;
	
	protected static final int matBRows = matrixSize;
	protected static final int matBColumns = matrixSize;
	
	public static void main(String[] args) throws Exception {
		FloatMatrixTest testSuit = new FloatMatrixTest();
		testSuit.setUp();
		testSuit.divColumnVectorTest();
	}
	
	protected Context context = Context.createOpenCLSinglePrecisionContext();
//	protected Context context = Context.createCudaSinglePrecisionContext();
//	protected Context context = Context.createJBLASSinglePrecisionContext();
	
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
	
//	@Test
//	public void repmatTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.repmat(1, 2);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = matA_GPU.repmat(1, 2);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
	
	@Test
	public void setSubMatrixTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.concatHorizontally(org.jblas.FloatMatrix.ones(matA_CPU.getRows(),1), matA_CPU);
		matC_CPU = org.jblas.FloatMatrix.concatVertically(org.jblas.FloatMatrix.ones(1, matC_CPU.getColumns()), matC_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrix.ones(matA_GPU.getRows()+1, matA_GPU.getColumns()+1, context);
		matC_GPU.setSubMatrix(matA_GPU, 1, 1);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void getSubMatrixTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.getRange(1, matA_CPU.getRows(), 1, matA_CPU.getColumns());
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.getSubMatrix(1, 1);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void addTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.add(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.add(matB_GPU);
				
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}

	@Test
	public void addScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.add(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.add(2);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}

	@Test
	public void addColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.ones(matA_CPU.getRows(), 1);
		org.jblas.FloatMatrix matC_CPU = matA_CPU.addColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix columnVector_GPU = FloatMatrix.ones(matA_GPU.getRows(), 1, context);
		FloatMatrix matC_GPU = matA_GPU.addColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		columnVector_GPU.free();
		matC_GPU.free();
	}
	
	@Test
	public void addRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.ones(1, matA_CPU.getColumns());
		org.jblas.FloatMatrix matC_CPU = matA_CPU.addRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix rowVector_GPU = FloatMatrix.ones(1, matA_GPU.getColumns(), context);
		FloatMatrix matC_GPU = matA_GPU.addRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		rowVector_GPU.free();
		matC_GPU.free();
	}
	
//	@Test
//	public void columnMaxsTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.columnMaxs();
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(1, matA_GPU.getColumns());
//		FloatMatrix.columnMaxs(matA_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void columnMaxsBadResultTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.columnMaxs();
//		
//		// Berechnung auf der GPU
//		// TODO es gibt keine Checks für falsche Dimensionen
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows() / 2, 1);
//		FloatMatrix.columnMaxs(matA_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void columnMeansTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.columnMeans();
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(1, matA_GPU.getColumns());
//		FloatMatrix.columnMeans(matA_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void columnMinsTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.columnMins();
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(1, matA_GPU.getColumns());
//		FloatMatrix.columnMins(matA_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void columnProdsTest() {
//		
//		// Berechnung auf der CPU
//		float[] matC_arr = new float[matA_CPU.getColumns()];
//		for (int c = 0; c < matA_CPU.getColumns(); c++)
//			matC_arr[c] = matA_CPU.getColumn(c).prod();
//		org.jblas.FloatMatrix matC_CPU = new org.jblas.FloatMatrix(1, matA_CPU.getColumns(), matC_arr);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(1, matA_GPU.getColumns());
//		FloatMatrix.columnProds(matA_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void columnSumsTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.columnSums();
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = matA_GPU.columnSums();
//		
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
	
	@Test
	public void divTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.div(matA_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.div(matA_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void divScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.div(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.div(2);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void divColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.ones(matA_CPU.getRows(), 1).muli(2);
		org.jblas.FloatMatrix matC_CPU = matA_CPU.divColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix columnVector_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, columnVector_CPU.toArray(), context);
		FloatMatrix matC_GPU = matA_GPU.divColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		columnVector_GPU.free();
		matC_GPU.free();
	}
	
	@Test
	public void divRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.ones(1, matA_CPU.getColumns()).muli(2);
		org.jblas.FloatMatrix matC_CPU = matA_CPU.divRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix rowVector_GPU = FloatMatrix.create(1, matA_GPU.getColumns(), rowVector_CPU.toArray(), context);
		FloatMatrix matC_GPU = matA_GPU.divRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		rowVector_GPU.free();
		matC_GPU.free();
	}
	

	
	
//	@Test
//	public void gtTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.gt(matB_CPU);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		matA_GPU.gt(matA_GPU, matB_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//
//	@Test
//	public void gtScalarTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.gt(0);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = matA_GPU.gt(0);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//
//	@Test
//	public void gtColumnVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getRows(); c++)
//			matC_CPU.putRow(c, matA_CPU.getRow(c).gt(columnVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix columnVector_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.gtColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		columnVector_GPU.free();
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void gtRowVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getColumns(); c++)
//			matC_CPU.putColumn(c, matA_CPU.getColumn(c).gt(rowVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix rowVector_GPU = FloatMatrix.create(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.gtRowVector(matA_GPU, rowVector_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		rowVector_GPU.free();
//		matC_GPU.free();
//	}
//	
//	
//	@Test
//	public void geTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.ge(matB_CPU);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.ge(matA_GPU, matB_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//
//	@Test
//	public void geScalarTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.ge(0.5f);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.ge(matA_GPU, 0.5f, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//
//	@Test
//	public void geColumnVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getRows(); c++)
//			matC_CPU.putRow(c, matA_CPU.getRow(c).ge(columnVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix columnVector_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.geColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		columnVector_GPU.free();
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void geRowVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getColumns(); c++)
//			matC_CPU.putColumn(c, matA_CPU.getColumn(c).ge(rowVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix rowVector_GPU = FloatMatrix.create(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.geRowVector(matA_GPU, rowVector_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		rowVector_GPU.free();
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void ltTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.lt(matB_CPU);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.lt(matA_GPU, matB_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//
//	@Test
//	public void ltScalarTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.lt(0.5f);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.lt(0.5f);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}

//	@Test
//	public void ltColumnVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getRows(); c++)
//			matC_CPU.putRow(c, matA_CPU.getRow(c).lt(columnVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix columnVector_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.ltColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		columnVector_GPU.free();
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void ltRowVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getColumns(); c++)
//			matC_CPU.putColumn(c, matA_CPU.getColumn(c).lt(rowVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix rowVector_GPU = FloatMatrix.create(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.ltRowVector(matA_GPU, rowVector_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		rowVector_GPU.free();
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void leTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.le(matB_CPU);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = matA_GPU.le(matB_GPU);		
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//
//	@Test
//	public void leScalarTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.le(0.5f);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = matA_GPU.le(0.5f);		
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//
//	@Test
//	public void leColumnVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getRows(); c++)
//			matC_CPU.putRow(c, matA_CPU.getRow(c).le(columnVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix columnVector_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.leColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		columnVector_GPU.free();
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void leRowVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getColumns(); c++)
//			matC_CPU.putColumn(c, matA_CPU.getColumn(c).le(rowVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix rowVector_GPU = FloatMatrix.create(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns());
//		FloatMatrix.leRowVector(matA_GPU, rowVector_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		rowVector_GPU.free();
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void eqTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.eq(matB_CPU);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = matA_GPU.eq(matB_GPU);		
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//
//	@Test
//	public void eqScalarTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.eq(0.5f);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = matA_GPU.eq(0.5f);		
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//
//	@Test
//	public void eqColumnVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getRows(); c++)
//			matC_CPU.putRow(c, matA_CPU.getRow(c).eq(columnVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix columnVector_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, columnVector_CPU.toArray(), context);
//		FloatMatrix matC_GPU = matA_GPU.eqColumnVector(columnVector_GPU);		
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		columnVector_GPU.free();
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void eqRowVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getColumns(); c++)
//			matC_CPU.putColumn(c, matA_CPU.getColumn(c).eq(rowVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix rowVector_GPU = FloatMatrix.create(1, matA_GPU.getColumns(), rowVector_CPU.toArray(), context);
//		FloatMatrix matC_GPU = matA_GPU.eqRowVector(rowVector_GPU);		
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		rowVector_GPU.free();
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void neTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.ne(matB_CPU);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = matA_GPU.ne(matB_GPU);		
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//
//	@Test
//	public void neScalarTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.ne(0.5f);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = matA_GPU.ne(0.5f);		
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//
//	@Test
//	public void neColumnVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getRows(); c++)
//			matC_CPU.putRow(c, matA_CPU.getRow(c).ne(columnVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix columnVector_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, columnVector_CPU.toArray(), context);
//		FloatMatrix matC_GPU = matA_GPU.neColumnVector(columnVector_GPU);		
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		columnVector_GPU.free();
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void neRowVectorTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//		for (int c = 0; c < matA_CPU.getColumns(); c++)
//			matC_CPU.putColumn(c, matA_CPU.getColumn(c).ne(rowVector_CPU));
//		
//		// Berechnung auf der GPU
//		FloatMatrix rowVector_GPU = FloatMatrix.create(1, matA_GPU.getColumns(), rowVector_CPU.toArray(), context);
//		FloatMatrix matC_GPU = matA_GPU.neRowVector(rowVector_GPU);		
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		rowVector_GPU.free();
//		matC_GPU.free();
//	}
	
	@Test
	public void mmulTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mmul(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mmul(matB_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	
	@Test
	public void mmulTransposeATest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.transpose().mmul(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mmulTN(matB_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void mmulTransposeBTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mmul(matB_CPU.transpose());
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mmulNT(matB_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}

	
	@Test
	public void mulTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mul(matA_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mul(matA_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void mulScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mul(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mul(2);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void mulColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.ones(matA_CPU.getRows(), 1).muli(2);
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mulColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix columnVector_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, columnVector_CPU.toArray(), context);
		FloatMatrix matC_GPU = matA_GPU.mulColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		columnVector_GPU.free();
		matC_GPU.free();
	}
	
	@Test
	public void mulRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.ones(1, matA_CPU.getColumns()).muli(2);
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mulRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix rowVector_GPU = FloatMatrix.create(1, matA_GPU.getColumns(), rowVector_CPU.toArray(), context);
		FloatMatrix matC_GPU = matA_GPU.mulRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		rowVector_GPU.free();
		matC_GPU.free();
	}

	@Test
	public void onesTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.ones(matA_CPU.getRows(), matB_CPU.getColumns());
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrix.ones(matA_GPU.getRows(), matB_GPU.getColumns(), context);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void rdivTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rdiv(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.rdiv(2);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}

	@Test
	public void rdivColumnVectorTest() {
		
		// Vorbereitungen
		float[] columnVector_arr = new float[matA_CPU.getRows()];
		for (int i = 0; i < columnVector_arr.length; i++)
			columnVector_arr[i] = i;
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < columnVector_arr.length; r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).rdivi(columnVector_arr[r]));

		// Berechnung auf der GPU
		FloatMatrix columnVector_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, columnVector_arr, context);
		FloatMatrix matC_GPU = matA_GPU.rdivColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		columnVector_GPU.free();
		matC_GPU.free();
	}
	
	@Test
	public void rdivRowVectorTest() {
		
		// Vorbereitungen
		float[] rowVector_arr = new float[matA_CPU.getColumns()];
		for (int i = 0; i < rowVector_arr.length; i++)
			rowVector_arr[i] = i;
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < rowVector_arr.length; c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).rdivi(rowVector_arr[c]));

		// Berechnung auf der GPU
		FloatMatrix rowVector_GPU = FloatMatrix.create(1, matA_GPU.getColumns(), rowVector_arr, context);
		FloatMatrix matC_GPU = matA_GPU.rdivRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		rowVector_GPU.free();
		matC_GPU.free();
	}
	
//	@Test
//	public void rowMaxsTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.rowMaxs();
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, context);
////		FloatMatrix.rowMaxs(matA_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void rowMeansTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.rowMeans();
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, context);
////		FloatMatrix.rowMeans(matA_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void rowMinsTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.rowMins();
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, context);
////		FloatMatrix.rowMins(matA_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void rowProdsTest() {
//		
//		// Berechnung auf der CPU
//		float[] matC_arr = new float[matA_CPU.getRows()];
//		for (int r = 0; r < matA_CPU.getRows(); r++)
//			matC_arr[r] = matA_CPU.getRow(r).prod();
//		org.jblas.FloatMatrix matC_CPU = new org.jblas.FloatMatrix(matA_CPU.getRows(), 1, matC_arr);
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, context);
////		FloatMatrix.rowProds(matA_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
//	
//	@Test
//	public void rowSumsTest() {
//		
//		// Berechnung auf der CPU
//		org.jblas.FloatMatrix matC_CPU = matA_CPU.rowSums();
//		
//		// Berechnung auf der GPU
//		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, context);
////		FloatMatrix.rowSums(matA_GPU, matC_GPU);
//		
//		// Ergebnisse vergleichen 
//		float[] result_CPU = matC_CPU.toArray();
//		float[] result_GPU = matC_GPU.toArray();
//		
//		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//		
//		matC_GPU.free();
//	}
	
	@Test
	public void rsubTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rsub(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.rsub(2);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}

	@Test
	public void rsubColumnVectorTest() {
		
		// Vorbereitungen
		float[] columnVector_arr = new float[matA_CPU.getRows()];
		for (int i = 0; i < columnVector_arr.length; i++)
			columnVector_arr[i] = i;
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < columnVector_arr.length; c++)
			matC_CPU.putRow(c, matA_CPU.getRow(c).rsubi(columnVector_arr[c]));

		// Berechnung auf der GPU
		FloatMatrix columnVector_GPU = FloatMatrix.create(matA_GPU.getRows(), 1, columnVector_arr, context);
		FloatMatrix matC_GPU = matA_GPU.rsubColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		columnVector_GPU.free();
		matC_GPU.free();
	}
	
	@Test
	public void rsubRowVectorTest() {
		
		// Vorbereitungen
		float[] rowVector_arr = new float[matA_CPU.getColumns()];
		for (int i = 0; i < rowVector_arr.length; i++)
			rowVector_arr[i] = i;
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < rowVector_arr.length; c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).rsubi(rowVector_arr[c]));

		// Berechnung auf der GPU
		FloatMatrix rowVector_GPU = FloatMatrix.create(1, matA_GPU.getColumns(), rowVector_arr, context);
		FloatMatrix matC_GPU = matA_GPU.rsubRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		rowVector_GPU.free();
		matC_GPU.free();
	}
	
	@Test
	public void subTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.sub(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.sub(matB_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void subScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.sub(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.sub(2);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}

	@Test
	public void subColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.ones(matA_CPU.getRows(), 1);
		org.jblas.FloatMatrix matC_CPU = matA_CPU.subColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix columnVector_GPU = FloatMatrix.ones(matA_GPU.getRows(), 1, context);
		FloatMatrix matC_GPU = matA_GPU.subColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		columnVector_GPU.free();
		matC_GPU.free();
	}
	
	@Test
	public void subRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.ones(1, matA_CPU.getColumns());
		org.jblas.FloatMatrix matC_CPU = matA_CPU.subRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix rowVector_GPU = FloatMatrix.ones(1, matA_GPU.getColumns(), context);
		FloatMatrix matC_GPU = matA_GPU.subRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		rowVector_GPU.free();
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
		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns(), context);
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
		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns(), context);
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
		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.sigmoid(matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
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
	public void meanTest() {
		
		// Berechnung auf der CPU
		float mean_CPU = matA_CPU.mean();
		
		// Berechnung auf der GPU
		float mean_GPU = matA_GPU.mean();
		
		Assert.assertEquals(mean_CPU, mean_GPU, 0.1f);
	}
	
	@Test
	public void prodTest() {
		
		// Berechnung auf der CPU
		float prod_CPU = matA_CPU.prod();
		
		// Berechnung auf der GPU
		float prod_GPU = matA_GPU.prod();
		
		Assert.assertEquals(prod_CPU, prod_GPU, 0.1f);
	}
	
	@Test
	public void maxTest() {
		
		// Berechnung auf der CPU
		float max_CPU = matA_CPU.max();
		
		// Berechnung auf der GPU
		float max_GPU = matA_GPU.max();
		
		Assert.assertEquals(max_CPU, max_GPU, 0.1f);
	}
	
	@Test
	public void minTest() {
		
		// Berechnung auf der CPU
		float min_CPU = matA_CPU.min();
		
		// Berechnung auf der GPU
		float min_GPU = matA_GPU.min();
		
		Assert.assertEquals(min_CPU, min_GPU, 0.1f);
	}
	
	
	@Test
	public void transposeTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.transpose();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.transpose(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}	
	
	@Test
	public void zerosTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matB_CPU.getColumns());
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrix.create(matA_GPU.getRows(), matB_GPU.getColumns(), context);
		matC_GPU.setZero();
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void toArray2Test() {
		
		// Ergebnisse vergleichen 
		float[][] result_CPU = matA_CPU.toArray2();
		float[][] result_GPU = matA_GPU.toArray2();
		
		for (int i = 0; i < result_GPU.length; i++)			
			Assert.assertArrayEquals(result_CPU[i], result_GPU[i], 0.1f);
	}
	
	@Test
	public void toArrayTest() {
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matA_CPU.toArray();
		float[] result_GPU = matA_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
	}
}
