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
		testSuit.rdivColumnVectorTest();
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
	public void addTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.add(matB_CPU);
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.add(matA_GPU, matB_GPU, matC_GPU);
		
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
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.add(matA_GPU, 2, matC_GPU);
		
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
		CLFloatMatrix columnVector_GPU = CLFloatMatrix.ones(matA_GPU.getRows(), 1);
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.addColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
		
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
		CLFloatMatrix rowVector_GPU = CLFloatMatrix.ones(1, matA_GPU.getColumns());
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.addRowVector(matA_GPU, rowVector_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		rowVector_GPU.free();
		matC_GPU.free();
	}
	
	@Test
	public void columnMaxsTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.columnMaxs();
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(1, matA_GPU.getColumns());
		CLFloatMatrix.columnMaxs(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void columnMaxsBadResultTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.columnMaxs();
		
		// Berechnung auf der GPU
		// TODO es gibt keine Checks für falsche Dimensionen
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows() / 2, 1);
		CLFloatMatrix.columnMaxs(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void columnMeansTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.columnMeans();
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(1, matA_GPU.getColumns());
		CLFloatMatrix.columnMeans(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void columnMinsTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.columnMins();
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(1, matA_GPU.getColumns());
		CLFloatMatrix.columnMins(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void columnProdsTest() {
		
		// Berechnung auf der CPU
		float[] matC_arr = new float[matA_CPU.getColumns()];
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_arr[c] = matA_CPU.getColumn(c).prod();
		org.jblas.FloatMatrix matC_CPU = new org.jblas.FloatMatrix(1, matA_CPU.getColumns(), matC_arr);
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(1, matA_GPU.getColumns());
		CLFloatMatrix.columnProds(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void columnSumsTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.columnSums();
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(1, matA_GPU.getColumns());
		CLFloatMatrix.columnSums(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void divTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.div(matA_CPU);
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.div(matA_GPU, matA_GPU, matC_GPU);
		
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
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.div(matA_GPU, 2, matC_GPU);
		
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
		CLFloatMatrix columnVector_GPU = new CLFloatMatrix(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.divColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
		
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
		CLFloatMatrix rowVector_GPU = new CLFloatMatrix(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.divRowVector(matA_GPU, rowVector_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		rowVector_GPU.free();
		matC_GPU.free();
	}
	
	
	@Test
	public void maxTest() {
		
		// Berechnung auf der CPU
		float max_CPU = matA_CPU.max();
		
		// Berechnung auf der GPU
		float max_GPU = CLFloatMatrix.max(matA_GPU);
		
		Assert.assertEquals(max_CPU, max_GPU, 0.1f);
	}
	
	@Test
	public void meanTest() {
		
		// Berechnung auf der CPU
		float mean_CPU = matA_CPU.mean();
		
		// Berechnung auf der GPU
		float mean_GPU = CLFloatMatrix.mean(matA_GPU);
		
		Assert.assertEquals(mean_CPU, mean_GPU, 0.1f);
	}
	
	@Test
	public void minTest() {
		
		// Berechnung auf der CPU
		float min_CPU = matA_CPU.min();
		
		// Berechnung auf der GPU
		float min_GPU = CLFloatMatrix.min(matA_GPU);
		
		Assert.assertEquals(min_CPU, min_GPU, 0.1f);
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
		
		matC_GPU.free();
	}
	
	
	@Test
	public void mmulTransposeATest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.transpose().mmul(matB_CPU);
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matB_GPU.getColumns());
		CLFloatMatrix.mmulTransposeA(matA_GPU, matB_GPU, matC_GPU);
		
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
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matB_GPU.getColumns());
		CLFloatMatrix.mmulTransposeB(matA_GPU, matB_GPU, matC_GPU);
		
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
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.mul(matA_GPU, matA_GPU, matC_GPU);
		
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
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.mul(matA_GPU, 2, matC_GPU);
		
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
		CLFloatMatrix columnVector_GPU = new CLFloatMatrix(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.mulColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
		
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
		CLFloatMatrix rowVector_GPU = new CLFloatMatrix(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.mulRowVector(matA_GPU, rowVector_GPU, matC_GPU);
		
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
		CLFloatMatrix matC_GPU = CLFloatMatrix.ones(matA_GPU.getRows(), matB_GPU.getColumns());
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void prodTest() {
		
		// Berechnung auf der CPU
		float prod_CPU = matA_CPU.prod();
		
		// Berechnung auf der GPU
		float prod_GPU = CLFloatMatrix.prod(matA_GPU);
		
		Assert.assertEquals(prod_CPU, prod_GPU, 0.1f);
	}
	
	@Test
	public void rdivTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rdiv(2);
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.rdiv(matA_GPU, 2, matC_GPU);
		
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
		for (int c = 0; c < columnVector_arr.length; c++)
			matC_CPU.putRow(c, matA_CPU.getRow(c).rdivi(columnVector_arr[c]));

		// Berechnung auf der GPU
		CLFloatMatrix columnVector_GPU = new CLFloatMatrix(matA_GPU.getRows(), 1, columnVector_arr);
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.rdivColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
		
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
		CLFloatMatrix rowVector_GPU = new CLFloatMatrix(1, matA_GPU.getColumns(), rowVector_arr);
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.rdivRowVector(matA_GPU, rowVector_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		rowVector_GPU.free();
		matC_GPU.free();
	}
	
	@Test
	public void rowMaxsTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rowMaxs();
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), 1);
		CLFloatMatrix.rowMaxs(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void rowMeansTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rowMeans();
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), 1);
		CLFloatMatrix.rowMeans(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void rowMinsTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rowMins();
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), 1);
		CLFloatMatrix.rowMins(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void rowProdsTest() {
		
		// Berechnung auf der CPU
		float[] matC_arr = new float[matA_CPU.getRows()];
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_arr[r] = matA_CPU.getRow(r).prod();
		org.jblas.FloatMatrix matC_CPU = new org.jblas.FloatMatrix(matA_CPU.getRows(), 1, matC_arr);
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), 1);
		CLFloatMatrix.rowProds(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void rowSumsTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rowSums();
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), 1);
		CLFloatMatrix.rowSums(matA_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
	@Test
	public void rsubTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rsub(2);
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.rsub(matA_GPU, 2, matC_GPU);
		
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
		CLFloatMatrix columnVector_GPU = new CLFloatMatrix(matA_GPU.getRows(), 1, columnVector_arr);
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.rsubColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
		
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
		CLFloatMatrix rowVector_GPU = new CLFloatMatrix(1, matA_GPU.getColumns(), rowVector_arr);
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.rsubRowVector(matA_GPU, rowVector_GPU, matC_GPU);
		
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
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.sub(matA_GPU, matB_GPU, matC_GPU);
		
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
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.sub(matA_GPU, 2, matC_GPU);
		
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
		CLFloatMatrix columnVector_GPU = CLFloatMatrix.ones(matA_GPU.getRows(), 1);
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.subColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
		
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
		CLFloatMatrix rowVector_GPU = CLFloatMatrix.ones(1, matA_GPU.getColumns());
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.subRowVector(matA_GPU, rowVector_GPU, matC_GPU);
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		rowVector_GPU.free();
		matC_GPU.free();
	}
	
	@Test
	public void sumTest() {
		
		// Berechnung auf der CPU
		float prod_CPU = matA_CPU.sum();
		
		// Berechnung auf der GPU
		float prod_GPU = CLFloatMatrix.sum(matA_GPU);
		
		Assert.assertEquals(prod_CPU, prod_GPU, 0.1f);
	}
	
	@Test
	public void transposeTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.transpose();
		
		// Berechnung auf der GPU
		CLFloatMatrix matC_GPU = new CLFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
		CLFloatMatrix.transpose(matA_GPU, matC_GPU);
		
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
		CLFloatMatrix matC_GPU = CLFloatMatrix.zeros(matA_GPU.getRows(), matB_GPU.getColumns());
		
		// Ergebnisse vergleichen 
		float[] result_CPU = matC_CPU.toArray();
		float[] result_GPU = matC_GPU.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		matC_GPU.free();
	}
	
}
