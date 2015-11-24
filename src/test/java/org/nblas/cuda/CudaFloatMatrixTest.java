package org.nblas.cuda;

import org.jblas.MatrixFunctions;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nblas.cuda.CudaFloatMatrix;
import org.nblas.cl.blas.CLLevel1;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.Random;

import org.jblas.MatrixFunctions;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class CudaFloatMatrixTest {

    protected static final int seed = 7;
    protected static final int runs = 100_000;
    protected static final int matrixSize = 251;

    protected static final int matARows = matrixSize;
    protected static final int matAColumns = matrixSize;

    protected static final int matBRows = matrixSize;
    protected static final int matBColumns = matrixSize;

    public static void main(String[] args) throws Exception {
        CudaFloatMatrixTest testSuit = new CudaFloatMatrixTest();
        testSuit.setUp();
        testSuit.zerosTest();
    }

    protected org.jblas.FloatMatrix matA_CPU;
    protected org.jblas.FloatMatrix matB_CPU;

    protected CudaFloatMatrix matA_GPU;
    protected CudaFloatMatrix matB_GPU;

    @Before
    public void setUp() throws Exception {
        Random rnd = new Random(seed);

        // Test-Daten anlegen
        float[] matAFloatArray = new float[matARows*matAColumns];
        float[] matBFloatArray = new float[matBRows*matBColumns];

        // Arrays mit Zufallszahlen füllen
        for (int i = 0; i < matAFloatArray.length; i++)
            matAFloatArray[i] = rnd.nextFloat() * 2 - 1;

        for (int i = 0; i < matBFloatArray.length; i++)
            matBFloatArray[i] = rnd.nextFloat() * 2 - 1;

        // die Daten auf die Grafikkarte kopieren
        matA_CPU = new org.jblas.FloatMatrix(matARows, matAColumns, matAFloatArray);
        matA_GPU = new CudaFloatMatrix(matARows, matAColumns, matAFloatArray);

        matB_CPU = new org.jblas.FloatMatrix(matBRows, matBColumns, matBFloatArray);
        matB_GPU = new CudaFloatMatrix(matBRows, matBColumns, matBFloatArray);
    }

    @After
    public void release(){
        matA_GPU.free();
        matB_GPU.free();
    }

    @Test
    public void memoryLeakTest() {

        CudaFloatMatrix ones_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        ones_GPU.setOne();
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matC_GPU.setZero();

        Instant start = Instant.now();
        for (int i = 0; i < runs; i++)
            matC_GPU.add(matC_GPU, ones_GPU, matC_GPU);
        System.out.println("took "+ Duration.between(start, Instant.now()));

        // überprüfe die Richtigkeit
        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.ones(matA_CPU.getRows(), matA_CPU.getColumns()).muli(runs);

        float[] result_CPU = matC_CPU.toArray();
        float[] result_GPU = matC_GPU.toArray();

        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);

        matC_GPU.free();
        ones_GPU.free();
    }

//    @Test
//    public void repmatTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.repmat(1, 2);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = matA_GPU.repmat(1, 2);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

    @Test
    public void setSubMatrixTest() {

        // Berechnung auf der CPU
        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.concatHorizontally(org.jblas.FloatMatrix.ones(matA_CPU.getRows(),1), matA_CPU);
        matC_CPU = org.jblas.FloatMatrix.concatVertically(org.jblas.FloatMatrix.ones(1, matC_CPU.getColumns()), matC_CPU);

        // Berechnung auf der GPU
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows()+1, matA_GPU.getColumns()+1);
        matC_GPU.setOne();
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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows()-1, matA_GPU.getColumns()-1);
        matA_GPU.getSubMatrix(matC_GPU, 1, 1);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.add(matA_GPU, matB_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.add(matA_GPU, 2, matC_GPU);

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
        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1);
        columnVector_GPU.setOne();
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.addColumnVector(matA_GPU, columnVector_GPU, matC_GPU);

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
        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns());
        rowVector_GPU.setOne();
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.addRowVector(matA_GPU, rowVector_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns());
        matA_GPU.columnMaxs(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows() / 2, 1);
        CudaFloatMatrix.columnMaxs(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns());
        CudaFloatMatrix.columnMeans(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns());
        CudaFloatMatrix.columnMins(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns());
        CudaFloatMatrix.columnProds(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns());
        CudaFloatMatrix.columnSums(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.div(matA_GPU, matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.div(matA_GPU, 2, matC_GPU);

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
        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        CudaFloatMatrix.divColumnVector(matA_GPU, columnVector_GPU, matC_GPU);

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
        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        CudaFloatMatrix.divRowVector(matA_GPU, rowVector_GPU, matC_GPU);

        // Ergebnisse vergleichen
        float[] result_CPU = matC_CPU.toArray();
        float[] result_GPU = matC_GPU.toArray();

        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);

        rowVector_GPU.free();
        matC_GPU.free();
    }




//    @Test
//    public void gtTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.gt(matB_CPU);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        matA_GPU.gt(matA_GPU, matB_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void gtScalarTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.gt(0);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        matA_GPU.gt(matA_GPU, 0, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void gtColumnVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getRows(); c++)
//            matC_CPU.putRow(c, matA_CPU.getRow(c).gt(columnVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.gtColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        columnVector_GPU.free();
//        matC_GPU.free();
//    }

//    @Test
//    public void gtRowVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getColumns(); c++)
//            matC_CPU.putColumn(c, matA_CPU.getColumn(c).gt(rowVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.gtRowVector(matA_GPU, rowVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        rowVector_GPU.free();
//        matC_GPU.free();
//    }


//    @Test
//    public void geTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.ge(matB_CPU);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.ge(matA_GPU, matB_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void geScalarTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.ge(0.5f);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.ge(matA_GPU, 0.5f, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void geColumnVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getRows(); c++)
//            matC_CPU.putRow(c, matA_CPU.getRow(c).ge(columnVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.geColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        columnVector_GPU.free();
//        matC_GPU.free();
//    }

//    @Test
//    public void geRowVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getColumns(); c++)
//            matC_CPU.putColumn(c, matA_CPU.getColumn(c).ge(rowVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.geRowVector(matA_GPU, rowVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        rowVector_GPU.free();
//        matC_GPU.free();
//    }

//    @Test
//    public void ltTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.lt(matB_CPU);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.lt(matA_GPU, matB_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void ltScalarTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.lt(0.5f);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.lt(matA_GPU, 0.5f, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void ltColumnVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getRows(); c++)
//            matC_CPU.putRow(c, matA_CPU.getRow(c).lt(columnVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.ltColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        columnVector_GPU.free();
//        matC_GPU.free();
//    }

//    @Test
//    public void ltRowVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getColumns(); c++)
//            matC_CPU.putColumn(c, matA_CPU.getColumn(c).lt(rowVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.ltRowVector(matA_GPU, rowVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        rowVector_GPU.free();
//        matC_GPU.free();
//    }

//    @Test
//    public void leTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.le(matB_CPU);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.le(matA_GPU, matB_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void leScalarTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.le(0.5f);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.le(matA_GPU, 0.5f, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void leColumnVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getRows(); c++)
//            matC_CPU.putRow(c, matA_CPU.getRow(c).le(columnVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.leColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        columnVector_GPU.free();
//        matC_GPU.free();
//    }

//    @Test
//    public void leRowVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getColumns(); c++)
//            matC_CPU.putColumn(c, matA_CPU.getColumn(c).le(rowVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.leRowVector(matA_GPU, rowVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        rowVector_GPU.free();
//        matC_GPU.free();
//    }

//    @Test
//    public void eqTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.eq(matB_CPU);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.eq(matA_GPU, matB_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void eqScalarTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.eq(0.5f);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.eq(matA_GPU, 0.5f, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void eqColumnVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getRows(); c++)
//            matC_CPU.putRow(c, matA_CPU.getRow(c).eq(columnVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.eqColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        columnVector_GPU.free();
//        matC_GPU.free();
//    }

//    @Test
//    public void eqRowVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getColumns(); c++)
//            matC_CPU.putColumn(c, matA_CPU.getColumn(c).eq(rowVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.eqRowVector(matA_GPU, rowVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        rowVector_GPU.free();
//        matC_GPU.free();
//    }

//    @Test
//    public void neTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.ne(matB_CPU);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.ne(matA_GPU, matB_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void neScalarTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix matC_CPU = matA_CPU.ne(0.5f);
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.ne(matA_GPU, 0.5f, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        matC_GPU.free();
//    }

//    @Test
//    public void neColumnVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix columnVector_CPU = org.jblas.FloatMatrix.rand(matA_CPU.getRows(), 1);
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getRows(); c++)
//            matC_CPU.putRow(c, matA_CPU.getRow(c).ne(columnVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.neColumnVector(matA_GPU, columnVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        columnVector_GPU.free();
//        matC_GPU.free();
//    }

//    @Test
//    public void neRowVectorTest() {
//
//        // Berechnung auf der CPU
//        org.jblas.FloatMatrix rowVector_CPU = org.jblas.FloatMatrix.rand(1, matA_CPU.getColumns());
//        org.jblas.FloatMatrix matC_CPU = org.jblas.FloatMatrix.zeros(matA_CPU.getRows(), matA_CPU.getColumns());
//        for (int c = 0; c < matA_CPU.getColumns(); c++)
//            matC_CPU.putColumn(c, matA_CPU.getColumn(c).ne(rowVector_CPU));
//
//        // Berechnung auf der GPU
//        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
//        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
//        CudaFloatMatrix.neRowVector(matA_GPU, rowVector_GPU, matC_GPU);
//
//        // Ergebnisse vergleichen
//        float[] result_CPU = matC_CPU.toArray();
//        float[] result_GPU = matC_GPU.toArray();
//
//        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);
//
//        rowVector_GPU.free();
//        matC_GPU.free();
//    }

    @Test
    public void maxTest() {

        // Berechnung auf der CPU
        float max_CPU = matA_CPU.max();

        // Berechnung auf der GPU
        float max_GPU = matA_GPU.max(matA_GPU);

        Assert.assertEquals(max_CPU, max_GPU, 0.1f);
    }

    @Test
    public void meanTest() {

        // Berechnung auf der CPU
        float mean_CPU = matA_CPU.mean();

        // Berechnung auf der GPU
        float mean_GPU = matA_GPU.mean(matA_GPU);

        Assert.assertEquals(mean_CPU, mean_GPU, 0.1f);
    }

    @Test
    public void minTest() {

        // Berechnung auf der CPU
        float min_CPU = matA_CPU.min();

        // Berechnung auf der GPU
        float min_GPU = matA_GPU.min(matA_GPU);

        Assert.assertEquals(min_CPU, min_GPU, 0.1f);
    }

    @Test
    public void mmulTest() {

        // Berechnung auf der CPU
        org.jblas.FloatMatrix matC_CPU = matA_CPU.mmul(matB_CPU);

        // Berechnung auf der GPU
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matB_GPU.getColumns());
        matC_GPU.mmul(matA_GPU, matB_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matB_GPU.getColumns());
        matC_GPU.mmulTN(matA_GPU, matB_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matB_GPU.getColumns());
        matC_GPU.mmulNT(matA_GPU, matB_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.mul(matA_GPU, matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.mul(matA_GPU, 2, matC_GPU);

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
        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1, columnVector_CPU.toArray());
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        CudaFloatMatrix.mulColumnVector(matA_GPU, columnVector_GPU, matC_GPU);

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
        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns(), rowVector_CPU.toArray());
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        CudaFloatMatrix.mulRowVector(matA_GPU, rowVector_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matB_GPU.getColumns());
        matC_GPU.setOne();

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
        float prod_GPU = matA_GPU.prod(matA_GPU);

        Assert.assertEquals(prod_CPU, prod_GPU, 0.1f);
    }

    @Test
    public void rdivTest() {

        // Berechnung auf der CPU
        org.jblas.FloatMatrix matC_CPU = matA_CPU.rdiv(2);

        // Berechnung auf der GPU
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.rdiv(matA_GPU, 2, matC_GPU);

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
        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1, columnVector_arr);
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        CudaFloatMatrix.rdivColumnVector(matA_GPU, columnVector_GPU, matC_GPU);

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
        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns(), rowVector_arr);
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        CudaFloatMatrix.rdivRowVector(matA_GPU, rowVector_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1);
        CudaFloatMatrix.rowMaxs(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1);
        CudaFloatMatrix.rowMeans(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1);
        CudaFloatMatrix.rowMins(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1);
        CudaFloatMatrix.rowProds(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1);
        CudaFloatMatrix.rowSums(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        CudaFloatMatrix.rsub(matA_GPU, 2, matC_GPU);

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
        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1, columnVector_arr);
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        CudaFloatMatrix.rsubColumnVector(matA_GPU, columnVector_GPU, matC_GPU);

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
        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns(), rowVector_arr);
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        CudaFloatMatrix.rsubRowVector(matA_GPU, rowVector_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.sub(matA_GPU, matB_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.sub(matA_GPU, 2, matC_GPU);

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
        CudaFloatMatrix columnVector_GPU = new CudaFloatMatrix(matA_GPU.getRows(), 1);
        columnVector_GPU.setOne();
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        CudaFloatMatrix.subColumnVector(matA_GPU, columnVector_GPU, matC_GPU);

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
        CudaFloatMatrix rowVector_GPU = new CudaFloatMatrix(1, matA_GPU.getColumns());
        rowVector_GPU.setOne();
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        CudaFloatMatrix.subRowVector(matA_GPU, rowVector_GPU, matC_GPU);

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
        float sum_CPU = matA_CPU.sum();

        // Berechnung auf der GPU
        float sum_GPU = matA_GPU.sum(matA_GPU);

        Assert.assertEquals(sum_CPU, sum_GPU, 1.0f);
    }

    @Test
    public void expTest() {

        // Berechnung auf der CPU
        org.jblas.FloatMatrix matC_CPU = MatrixFunctions.exp(matA_CPU);

        // Berechnung auf der GPU
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.exp(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.neg(matA_GPU, matC_GPU);

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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
        matA_GPU.sigmoid(matA_GPU, matC_GPU);

        // Ergebnisse vergleichen
        float[] result_CPU = matC_CPU.toArray();
        float[] result_GPU = matC_GPU.toArray();

        Assert.assertArrayEquals(result_CPU, result_GPU, 0.1f);

        matC_GPU.free();
    }

    @Test
    public void transposeTest() {

        // Berechnung auf der CPU
        org.jblas.FloatMatrix matC_CPU = matA_CPU.transpose();

        // Berechnung auf der GPU
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matA_GPU.getColumns());
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
        CudaFloatMatrix matC_GPU = new CudaFloatMatrix(matA_GPU.getRows(), matB_GPU.getColumns());
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
