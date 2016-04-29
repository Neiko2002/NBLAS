package org.nblas;

import java.util.Random;

import org.jblas.MatrixFunctions;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nblas.impl.FloatMatrixDefault;

/**
 * operation prefix:
 * i = in place
 * o = to output matrix
 * 
 * @author Nico
 *
 */
public class FloatMatrixTest {

	protected static final int seed = 7;
	protected static final int runs = 100_000;
	protected static final int matrixRows = 80; 
	protected static final int matrixColumns = 100; 
	
	public static void main(String[] args) throws Exception {
		FloatMatrixTest testSuit = new FloatMatrixTest();
		testSuit.setUp();
		testSuit.transposeTest();
	}
	
	protected Context context = Context.OpenCLSinglePrecisionContext;
//	protected Context context = Context.CudaSinglePrecisionContext;
//	protected Context context = Context.JBLASSinglePrecisionContext;
	
	protected org.jblas.FloatMatrix matA_CPU;
	protected org.jblas.FloatMatrix matB_CPU;
	
	protected org.jblas.FloatMatrix rowVector_CPU;
	protected org.jblas.FloatMatrix columnVector_CPU;
	
	protected FloatMatrix matA_GPU;
	protected FloatMatrix matB_GPU;
	
	protected FloatMatrix rowVector_GPU;
	protected FloatMatrix columnVector_GPU;

	@Before
	public void setUp() throws Exception {
		Random rnd = new Random(seed);
		
		// Test-Daten anlegen
		float[] matAFloatArray = new float[matrixRows*matrixColumns];
		float[] matBFloatArray = new float[matrixRows*matrixColumns];
		float[] rowVecFloatArray = new float[matrixColumns];
		float[] colVecFloatArray = new float[matrixRows];
		
		// Arrays mit Zufallszahlen füllen
		for (int i = 0; i < matAFloatArray.length; i++) 
			matAFloatArray[i] = rnd.nextFloat();

		for (int i = 0; i < matBFloatArray.length; i++) 
			matBFloatArray[i] = rnd.nextFloat();
		
		for (int i = 0; i < rowVecFloatArray.length; i++) 
			rowVecFloatArray[i] = rnd.nextFloat();
		
		for (int i = 0; i < colVecFloatArray.length; i++) 
			colVecFloatArray[i] = rnd.nextFloat();
		
		// die Daten auf die Grafikkarte kopieren
		matA_CPU = new org.jblas.FloatMatrix(matrixRows, matrixColumns, matAFloatArray.clone());
		matA_GPU = FloatMatrix.create(matrixRows, matrixColumns, matAFloatArray.clone(), context);
		
		matB_CPU = new org.jblas.FloatMatrix(matrixRows, matrixColumns, matBFloatArray.clone());
		matB_GPU = FloatMatrix.create(matrixRows, matrixColumns, matBFloatArray.clone(), context);
		
		rowVector_CPU = new org.jblas.FloatMatrix(1, matrixColumns, rowVecFloatArray.clone());
		rowVector_GPU = FloatMatrix.create(1, matrixColumns, rowVecFloatArray.clone(), context);
		
		columnVector_CPU = new org.jblas.FloatMatrix(matrixRows, 1, colVecFloatArray.clone());
		columnVector_GPU = FloatMatrix.create(matrixRows, 1, colVecFloatArray.clone(), context);
	}

	@After
	public void release() throws Exception {
		matA_GPU.release();
		matB_GPU.release();
	}
	
	/**
	 * Compares the content of the two matricies jblas and nblas and free the nblas resource as well as all other matricies.
	 * 
	 * @param jblasMat
	 * @param nblasMat
	 * @param other
	 */
		
	protected void assertAndFree(org.jblas.FloatMatrix jblasMat, FloatMatrix nblasMat, FloatMatrix ... other) {
		// Ergebnisse vergleichen 
		float[] result_CPU = jblasMat.toArray();
		float[] result_GPU = nblasMat.toArray();
		
		Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
		
		// free the resources
		nblasMat.release();
		for (FloatMatrix mat : other)
			mat.release();
	}

	
	@Test
	public void duplicateTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void repmatTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.repmat(1, 2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.repmat(1, 2);
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- add tests --------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void addTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.add(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.add(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void addiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.add(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.addi(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void addoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.add(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.add(matB_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void addScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.add(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.add(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void addiScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().addi(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().addi(2);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void addoScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().addi(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.add(2, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void addColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.addColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.addColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void addiColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.addColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.addiColumnVector(columnVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void addoColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.addColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.addColumnVector(columnVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void addRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.addRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.addRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void addiRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.addRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.addiRowVector(rowVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void addoRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.addRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.addRowVector(rowVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	
	
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- sub tests --------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void subTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.sub(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.sub(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void subiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.sub(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().subi(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void suboTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.sub(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.sub(matB_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void subScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.sub(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.sub(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void subiScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().subi(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().subi(2);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void suboScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().subi(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.sub(2, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void subColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.subColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.subColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void subiColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.subColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.subiColumnVector(columnVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void suboColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.subColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.subColumnVector(columnVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void subRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.subRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.subRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	@Test
	public void subiRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.subRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.subiRowVector(rowVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void suboRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.subRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.subRowVector(rowVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rsubTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rsub(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.rsub(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void rsubiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rsub(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().rsubi(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rsuboTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rsub(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.rsub(2, matC_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rsubColumnVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < columnVector_CPU.data.length; c++)
			matC_CPU.putRow(c, matA_CPU.getRow(c).rsubi(columnVector_CPU.data[c]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.rsubColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rsubiColumnVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < columnVector_CPU.data.length; c++)
			matC_CPU.putRow(c, matA_CPU.getRow(c).rsubi(columnVector_CPU.data[c]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().rsubiColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rsuboColumnVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < columnVector_CPU.data.length; c++)
			matC_CPU.putRow(c, matA_CPU.getRow(c).rsubi(columnVector_CPU.data[c]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.rsubColumnVector(columnVector_GPU, matC_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rsubRowVectorTest() {
	
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < rowVector_CPU.data.length; c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).rsubi(rowVector_CPU.data[c]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.rsubRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rsubiRowVectorTest() {
	
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < rowVector_CPU.data.length; c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).rsubi(rowVector_CPU.data[c]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().rsubiRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rsuboRowVectorTest() {
	
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < rowVector_CPU.data.length; c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).rsubi(rowVector_CPU.data[c]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.rsubRowVector(rowVector_GPU, matC_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- mul tests --------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void mulTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mul(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mul(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void muliTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mul(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.muli(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void muloTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mul(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.mul(matB_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void mulScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mul(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mul(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void muliScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().muli(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().muli(2);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void muloScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().muli(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.mul(2, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void mulColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mulColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mulColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void muliColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mulColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.muliColumnVector(columnVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void muloColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mulColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.mulColumnVector(columnVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void mulRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mulRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mulRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	@Test
	public void muliRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mulRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.muliRowVector(rowVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void muloRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mulRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.mulRowVector(rowVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	
	
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- div tests --------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void divTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.div(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.div(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void diviTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.div(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().divi(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void divoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.div(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.div(matB_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void divScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.div(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.div(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void diviScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().divi(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().divi(2);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void divoScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().divi(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.div(2, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void divColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.divColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.divColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void diviColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.divColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().diviColumnVector(columnVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void divoColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.divColumnVector(columnVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.divColumnVector(columnVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void divRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.divRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.divRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	@Test
	public void diviRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.divRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().diviRowVector(rowVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void divoRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.divRowVector(rowVector_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.divRowVector(rowVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
		
	@Test
	public void rdivTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rdiv(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.rdiv(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rdiviTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rdiv(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().rdivi(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rdivoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.rdiv(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.rdiv(2, matC_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void rdivColumnVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < columnVector_CPU.data.length; r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).rdivi(columnVector_CPU.data[r]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.rdivColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rdiviColumnVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < columnVector_CPU.data.length; r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).rdivi(columnVector_CPU.data[r]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().rdivColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rdivoColumnVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < columnVector_CPU.data.length; r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).rdivi(columnVector_CPU.data[r]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.rdivColumnVector(columnVector_GPU, matC_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rdivRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < rowVector_CPU.data.length; c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).rdivi(rowVector_CPU.data[c]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.rdivRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rdiviRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < rowVector_CPU.data.length; c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).rdivi(rowVector_CPU.data[c]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().rdiviRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void rdivoRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < rowVector_CPU.data.length; c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).rdivi(rowVector_CPU.data[c]));

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.rdivRowVector(rowVector_GPU, matC_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- gt tests ---------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void gtTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.gt(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.gt(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void gtiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.gt(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.gti(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void gtoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.gt(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.gt(matB_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void gtScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.gt(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.gt(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void gtiScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().gti(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().gti(2);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void gtoScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().gti(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.gt(2, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void gtColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).gt(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.gtColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void gtiColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).gt(columnVector_CPU));	
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.gtiColumnVector(columnVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void gtoColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).gt(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.gtColumnVector(columnVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void gtRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).gt(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.gtRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void gtiRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).gt(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.gtiRowVector(rowVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void gtoRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).gt(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.gtRowVector(rowVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	

    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- ge tests ---------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void geTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.ge(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.ge(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void geiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.ge(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.gei(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void geoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.ge(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.ge(matB_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void geScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.ge(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.ge(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void geiScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().gei(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().gei(2);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void geoScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().gei(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.ge(2, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void geColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).ge(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.geColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void geiColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).ge(columnVector_CPU));	
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.geiColumnVector(columnVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void geoColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).ge(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.geColumnVector(columnVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void geRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).ge(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.geRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	@Test
	public void geiRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).ge(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.geiRowVector(rowVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void geoRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).ge(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.geRowVector(rowVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	

    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- lt tests ---------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void ltTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.lt(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.lt(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void ltiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.lt(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().lti(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void ltoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.lt(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.lt(matB_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void ltScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.lt(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.lt(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void ltiScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().lti(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().lti(2);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void ltoScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().lti(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.lt(2, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void ltColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).lt(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.ltColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void ltiColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).lt(columnVector_CPU));	
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.ltiColumnVector(columnVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void ltoColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).lt(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.ltColumnVector(columnVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void ltRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).lt(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.ltRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	@Test
	public void ltiRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).lt(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.ltiRowVector(rowVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void ltoRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).lt(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.ltRowVector(rowVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}


    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- le tests ---------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void leTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.le(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.le(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void leiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.le(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.lei(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void leoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.le(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.le(matB_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void leScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.le(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.le(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void leiScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().lei(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().lei(2);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void leoScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().lei(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.le(2, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void leColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).le(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.leColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void leiColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).le(columnVector_CPU));	
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.leiColumnVector(columnVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void leoColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).le(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.leColumnVector(columnVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void leRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).le(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.leRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	@Test
	public void leiRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).le(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.leiRowVector(rowVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void leoRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).le(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.leRowVector(rowVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	

    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- eq tests ---------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void eqTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.eq(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.eq(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void eqiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.eq(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.eqi(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void eqoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.eq(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.eq(matB_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void eqScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.eq(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.eq(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void eqiScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().eqi(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().eqi(2);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void eqoScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().eqi(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.eq(2, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void eqColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).eq(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.eqColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void eqiColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).eq(columnVector_CPU));	
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.eqiColumnVector(columnVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void eqoColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).eq(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.eqColumnVector(columnVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void eqRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).eq(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.eqRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	@Test
	public void eqiRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).eq(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.eqiRowVector(rowVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void eqoRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).eq(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.eqRowVector(rowVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	

    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- ne tests ---------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void neTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.ne(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.ne(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void neiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.ne(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.nei(matB_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void neoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.ne(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.ne(matB_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void neScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.ne(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.ne(2);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void neiScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().nei(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().nei(2);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void neoScalarTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().nei(2);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.ne(2, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void neColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).ne(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.neColumnVector(columnVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void neiColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).ne(columnVector_CPU));	
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.neiColumnVector(columnVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void neoColumnVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int c = 0; c < matA_CPU.getColumns(); c++)
			matC_CPU.putColumn(c, matA_CPU.getColumn(c).ne(columnVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.neColumnVector(columnVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void neRowVectorTest() {
				
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).ne(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.neRowVector(rowVector_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	@Test
	public void neiRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).ne(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup();
		matC_GPU.neiRowVector(rowVector_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void neoRowVectorTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
		for (int r = 0; r < matA_CPU.getRows(); r++)
			matC_CPU.putRow(r, matA_CPU.getRow(r).ne(rowVector_CPU));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.neRowVector(rowVector_GPU, matC_GPU);
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------- matrix multiplication tests --------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	
	@Test
	public void mmulTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mmul(matB_CPU.transpose());
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mmul(matB_GPU.transpose());		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void mmuloTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mmul(matB_CPU.transpose());
		
		// Berechnung auf der GPU
    	FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matB_GPU.getRows(), matA_GPU.getContext());
		matA_GPU.mmul(matB_GPU.transpose(), matC_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	
	@Test
	public void mmulTNTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.transpose().mmul(matB_CPU);
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mmulTN(matB_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void mmulTNoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.transpose().mmul(matB_CPU);
		
		// Berechnung auf der GPU
    	FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getColumns(), matB_GPU.getColumns(), matA_GPU.getContext());
    	matA_GPU.mmulTN(matB_GPU, matC_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void mmulNTTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mmul(matB_CPU.transpose());
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.mmulNT(matB_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	@Test
	public void mmulNToTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.mmul(matB_CPU.transpose());
		
		// Berechnung auf der GPU
    	FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matB_GPU.getRows(), matA_GPU.getContext());
    	matA_GPU.mmulNT(matB_GPU, matC_GPU);		
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}

	
    // ----------------------------------------------------------------------------------------------------------
    // ------------------------------------------- exponential tests --------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void expTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = MatrixFunctions.exp(matA_CPU);

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.exp();
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void expiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = MatrixFunctions.exp(matA_CPU);

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().expi();
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void expoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = MatrixFunctions.exp(matA_CPU);

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.exp(matC_GPU);
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	
	
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- negation tests ---------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void negTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.neg();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.neg();
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	

	@Test
	public void negiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.neg();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().negi();
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void negoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.neg();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.neg(matC_GPU);
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	
	
	
    // ----------------------------------------------------------------------------------------------------------
    // ---------------------------------------------- sigmoid tests ---------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
	@Test
	public void sigmoidTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup();
		for (int i = 0; i < matA_CPU.data.length; i++)
			matC_CPU.data[i] = (float) (1. / ( 1. + Math.exp(-matA_CPU.data[i]) ));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.sigmoid();
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void sigmoidiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup();
		for (int i = 0; i < matA_CPU.data.length; i++)
			matC_CPU.data[i] = (float) (1. / ( 1. + Math.exp(-matA_CPU.data[i]) ));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().sigmoidi();
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void sigmoidoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup();
		for (int i = 0; i < matA_CPU.data.length; i++)
			matC_CPU.data[i] = (float) (1. / ( 1. + Math.exp(-matA_CPU.data[i]) ));
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matA_GPU.getColumns(), context);
		matA_GPU.sigmoid(matC_GPU);
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	
	
    // ----------------------------------------------------------------------------------------------------------
    // ------------------------------------------- aggregation tests --------------------------------------------
	// ----------------------------------------------------------------------------------------------------------
	
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
		
		// Ergebnisse vergleichen 
		Assert.assertEquals(mean_CPU, mean_GPU, 0.1f);
	}
	
	@Test
	public void prodTest() {
		
		// Berechnung auf der CPU
		float prod_CPU = matA_CPU.prod();
		
		// Berechnung auf der GPU
		float prod_GPU = matA_GPU.prod();
		
		// Ergebnisse vergleichen 
		Assert.assertEquals(prod_CPU, prod_GPU, 0.1f);
	}
	
	@Test
	public void maxTest() {
		
		// Berechnung auf der CPU
		float max_CPU = matA_CPU.max();
		
		// Berechnung auf der GPU
		float max_GPU = matA_GPU.max();
		
		// Ergebnisse vergleichen 
		Assert.assertEquals(max_CPU, max_GPU, 0.1f);
	}
	
	@Test
	public void minTest() {
		
		// Berechnung auf der CPU
		float min_CPU = matA_CPU.min();
		
		// Berechnung auf der GPU
		float min_GPU = matA_GPU.min();
		
		// Ergebnisse vergleichen 
		Assert.assertEquals(min_CPU, min_GPU, 0.1f);
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
	public void transposeoTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.transpose();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getColumns(), matA_GPU.getRows(), context);
		matA_GPU.transpose(matC_GPU);
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}	
	
	@Test
	public void transposeTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.transpose();
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.transpose();	
				
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}	
	
	

	
    // ----------------------------------------------------------------------------------------------------------
    // --------------------------------------------- manipulation tests -----------------------------------------
	// ----------------------------------------------------------------------------------------------------------

	
	@Test
	public void setSubMatrixTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.concatHorizontally(org.jblas.FloatMatrix.ones(matA_CPU.getRows(), 1), matA_CPU); // füge eine Spalte mit 1 hinzu
		matC_CPU = org.jblas.FloatMatrix.concatVertically(org.jblas.FloatMatrix.ones(1, matC_CPU.getColumns()), matC_CPU);						// füge eine Zeile mit 1 hinzu
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrix.ones(matA_GPU.getRows()+1, matA_GPU.getColumns()+1, context);
		matC_GPU.setSubMatrix(matA_GPU, 1, 1);
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void getSubMatrixTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.getRange(1, matA_CPU.getRows(), 1, matA_CPU.getColumns());
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.getSubMatrix(1, 1);
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void putTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup().put(matA_CPU.getRows()/2, matA_CPU.getColumns()/2, 1);

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().put(matA_GPU.getRows()/2, matA_GPU.getColumns()/2, 1);

		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);	
	}
	
	@Test
	public void getTest() {
		
		// Berechnung auf der CPU
		float val_CPU = matA_CPU.get(matA_CPU.getRows()/2, matA_CPU.getColumns()/2);

		// Berechnung auf der GPU
		float val_GPU = matA_GPU.get(matA_GPU.getRows()/2, matA_GPU.getColumns()/2);

		// Ergebnisse vergleichen 
		Assert.assertEquals(val_CPU, val_GPU, 0.1f);
	}
	
	@Test
	public void putRowTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup();
		matC_CPU.putRow(0, rowVector_CPU);

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().putRow(rowVector_GPU, 0);

		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);	
	}
	
	@Test
	public void getRowTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.getRow(0);

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.getRow(0);

		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);	
	}
	
	@Test
	public void getRowiTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.getRow(0, rowVector_CPU);
		

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.getRow(rowVector_GPU, 0);

		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);	
	}
	
	@Test
	public void putColumnTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.dup();
		matC_CPU.putColumn(0, columnVector_CPU);

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.dup().putColumn(columnVector_GPU, 0);

		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);	
	}
	
	@Test
	public void getColumnTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.getColumn(0);

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.getColumn(0);

		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);	
	}
	
	@Test
	public void getColumniTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = matA_CPU.getColumn(0, columnVector_CPU);
		

		// Berechnung auf der GPU
		FloatMatrix matC_GPU = matA_GPU.getColumn(columnVector_GPU, 0);

		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);	
	}
	
	@Test
	public void onesTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.ones(matA_CPU.getRows(), matB_CPU.getColumns());
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrix.ones(matA_GPU.getRows(), matB_GPU.getColumns(), context);
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
	}
	
	@Test
	public void zerosTest() {
		
		// Berechnung auf der CPU
		org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matB_CPU.getColumns());
		
		// Berechnung auf der GPU
		FloatMatrix matC_GPU = FloatMatrixDefault.dirtyAllocation(matA_GPU.getRows(), matB_GPU.getColumns(), context);
		matC_GPU.setZero();
		
		// Ergebnisse vergleichen 
		assertAndFree(matC_CPU, matC_GPU);
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
