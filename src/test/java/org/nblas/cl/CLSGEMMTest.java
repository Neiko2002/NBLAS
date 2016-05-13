package org.nblas.cl;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import org.jocl.CL;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;
import org.nblas.Context;
import org.nblas.FloatMatrix;
import org.nblas.cl.model.CLArray;
import org.nblas.cl.model.CLScalar;
import org.nblas.generic.Subprogram;

/**
 * http://www.cedricnugteren.nl/tutorial.php?page=3
 * 
 * @author Nico
 *
 */
public class CLSGEMMTest {

	protected CLCore CORE = CLCore.getCore();
	protected Context context = Context.OpenCLSinglePrecisionContext;
	protected Map<String, Subprogram<cl_kernel>> kernels = new HashMap<>();
	
	protected int TS = 32;		// Tile Size
	protected int WPT = 8;		// The amount of work-per-thread, i.e. the thread-coarsening factor
	protected int WIDTH = 4;	// Wider data-types float4
	
	protected int RTS = (TS/WPT);            // The reduced tile-size in one dimension
	protected int TSDK = 16;     			// The tile-size in dimension K (for kernel 5 only)
	protected int LPT = ((TSDK*WPT)/(TS));	// The amount of loads-per-thread (assume TSN==TSM)
	
	protected int TSM = 128;                // The tile-size in dimension M
	protected int TSN = 128;                // The tile-size in dimension N
	protected int TSK = 16;                 // The tile-size in dimension K
	protected int WPTM = 8;                 // The work-per-thread in dimension M
	protected int WPTN = 8;                 // The work-per-thread in dimension N
	protected int RTSM = (TSM/WPTM);        // The reduced tile-size in dimension M
	protected int RTSN = (TSN/WPTN);        // The reduced tile-size in dimension N
	protected int LPTA = ((TSK*TSM)/(RTSM*RTSN)); // Loads-per-thread for A
	protected int LPTB = ((TSK*TSN)/(RTSM*RTSN)); // Loads-per-thread for B
	
	protected int THREADSX = 8;
	protected int THREADSY = 8;
	protected int RX = 8;
	protected int RY = 4;
	
	protected int TRANSPOSEY = 16;
	protected int TRANSPOSEX = 16;
	
	// 256 512 1024 2048 4096
	protected int iterations = 10;
	protected int M = 2048; // matrix1 und matrix3 rows 
	protected int N = 2048; // matrix2 und matrix3 columns
	protected int K = 2048; // matrix1 column und matrix2 rows
	protected CLFloatMatrix matrix1;
	protected CLFloatMatrix matrix2;
	protected CLFloatMatrix matrix2T;
	protected CLFloatMatrix matrix3;
	protected long flopPerIteration;

	public static void main(String[] args) throws IOException, URISyntaxException {
		CLSGEMMTest testSuit = new CLSGEMMTest();
		testSuit.setUp();
		testSuit.oclMmul1();
		testSuit.oclMmul2();
		testSuit.SGEMM();
		testSuit.mySGEMM1();
		testSuit.mySGEMM2();
		testSuit.mySGEMM3();
		testSuit.mySGEMM4();
		testSuit.mySGEMM5();
		testSuit.mySGEMM6();
		testSuit.mySGEMM7();
		testSuit.mySGEMM11();
		testSuit.mySGEMM12();
	}
	
	protected void mySGEMM12() {
		cl_kernel kernel = kernels.get("mySGEMM12").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		

		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2.getSizeof(), matrix2.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, CLScalar.of(M).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, CLScalar.of(N).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(K).getPointer());
		CL.clSetKernelArg(kernel, 6, Sizeof.cl_float, CLScalar.of(1f).getPointer());
		CL.clSetKernelArg(kernel, 7, Sizeof.cl_float, CLScalar.of(1f).getPointer());
		CL.clSetKernelArg(kernel, 8, Sizeof.cl_uint, CLScalar.of(matrix1.clColumns).getPointer());
		CL.clSetKernelArg(kernel, 9, Sizeof.cl_uint, CLScalar.of(matrix2.clColumns).getPointer());
		CL.clSetKernelArg(kernel, 10, Sizeof.cl_uint, CLScalar.of(matrix3.clColumns).getPointer());
		
		for (int i = 0; i < iterations; i++)
			CORE.enqueue2DRangeKernelTest(kernel, N/RY, M/RX, THREADSY, THREADSX); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("mySGEMM12 took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);   
	}
	
	protected void mySGEMM11() {
		cl_kernel kernel = kernels.get("mySGEMM11").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		

		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2.getSizeof(), matrix2.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, CLScalar.of(M).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, CLScalar.of(N).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(K).getPointer());
		
		for (int i = 0; i < iterations; i++)
			CORE.enqueue2DRangeKernelTest(kernel, N/RY, M/RX, THREADSY, THREADSX); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("mySGEMM11 took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);   
	}
	
	protected void mySGEMM7() {
		cl_kernel kernel = kernels.get("mySGEMM7").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
		
		transpose();
		CORE.waitOnComplete();
		
		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2T.getSizeof(), matrix2T.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, CLScalar.of(M).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, CLScalar.of(N).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(K).getPointer());
		
		for (int i = 0; i < iterations; i++)
			CORE.enqueue2DRangeKernelTest(kernel, N/WPTN, M/WPTM, TSN/WPTN, TSM/WPTM); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("mySGEMM7 took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);   
	}
	
	protected void mySGEMM6() {
		cl_kernel kernel = kernels.get("mySGEMM6").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
		
		transpose();
		CORE.waitOnComplete();
		
		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2T.getSizeof(), matrix2T.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, CLScalar.of(M).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, CLScalar.of(N).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(K).getPointer());
		
		for (int i = 0; i < iterations; i++)
			CORE.enqueue2DRangeKernelTest(kernel, N/WPTN, M/WPTM, TSN/WPTN, TSM/WPTM); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("mySGEMM6 took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);   
	}
	
	protected void mySGEMM5() {
		cl_kernel kernel = kernels.get("mySGEMM5").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
		
		transpose();
		CORE.waitOnComplete();
		
		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2T.getSizeof(), matrix2T.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, CLScalar.of(M).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, CLScalar.of(N).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(K).getPointer());
		
		for (int i = 0; i < iterations; i++)
			CORE.enqueue2DRangeKernelTest(kernel, N/WPT, M, RTS, TS); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("mySGEMM5 took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);   
	}
	
	protected void mySGEMM4() {
		cl_kernel kernel = kernels.get("mySGEMM4").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
		    
		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2.getSizeof(), matrix2.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, CLScalar.of(M).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, CLScalar.of(N).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(K).getPointer());
		
		for (int i = 0; i < iterations; i++)	
		CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns/WIDTH, TS, TS/WIDTH); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("mySGEMM4 took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);  
	}
	
	protected void mySGEMM3() {
		cl_kernel kernel = kernels.get("mySGEMM3").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
		    
		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2.getSizeof(), matrix2.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, CLScalar.of(M).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, CLScalar.of(N).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(K).getPointer());
		
		for (int i = 0; i < iterations; i++)
			CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows/WPT, matrix2.clColumns, RTS, TS); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("mySGEMM3 took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);  
	}
	
	protected void mySGEMM2() {
		cl_kernel kernel = kernels.get("mySGEMM2").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
		    
		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2.getSizeof(), matrix2.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, CLScalar.of(M).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, CLScalar.of(N).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(K).getPointer());
		
		for (int i = 0; i < iterations; i++)
			CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, TS, TS); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("mySGEMM2 took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);  
	}
	
	protected void mySGEMM1() {
		cl_kernel kernel = kernels.get("mySGEMM1").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
		    
		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2.getSizeof(), matrix2.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, CLScalar.of(M).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, CLScalar.of(N).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(K).getPointer());
  
		for (int i = 0; i < iterations; i++)
			CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, TS, TS); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("mySGEMM1 took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);  
	}
	
	protected void oclMmul2() {
		cl_kernel kernel = kernels.get("oclMmul2").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
		    
		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2.getSizeof(), matrix2.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, CLScalar.of(matrix3.clRows).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_uint, CLScalar.of(matrix1.clColumns).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(matrix3.clColumns).getPointer());
		CL.clSetKernelArg(kernel, 6, Sizeof.cl_float*CORE.getThreadCount(), CLArray.ofFloat(CORE.getThreadCount()).getPointer());
		CL.clSetKernelArg(kernel, 7, Sizeof.cl_float*CORE.getThreadCount(), CLArray.ofFloat(CORE.getThreadCount()).getPointer());
  
		for (int i = 0; i < iterations; i++)
			CORE.enqueue2DRangeKernelTest(kernel, matrix3.clRows, matrix3.clColumns, TS, TS); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("oclMmul2 took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);   
	}
	
	protected void oclMmul1() {
		cl_kernel kernel = kernels.get("oclMmul1").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
		    
		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2.getSizeof(), matrix2.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_float*CORE.getThreadCount(), CLArray.ofFloat(CORE.getThreadCount()).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_float*CORE.getThreadCount(), CLArray.ofFloat(CORE.getThreadCount()).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(matrix1.clColumns).getPointer());
		CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, CLScalar.of(matrix2.clColumns).getPointer());
  
		for (int i = 0; i < iterations; i++)
			CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, TS, TS); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("oclMmul1 took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);    
	}
	
	protected void SGEMM() {
		cl_kernel kernel = CLPredefined.getSubprogram("sgemm_nn").getKernel();
				
		// führe den Kernel aus und messe die Zeit
		CORE.waitOnComplete();
		long start= System.currentTimeMillis();		
		    
		CL.clSetKernelArg(kernel, 0, matrix1.getSizeof(), matrix1.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2.getSizeof(), matrix2.getPointer());
		CL.clSetKernelArg(kernel, 2, matrix3.getSizeof(), matrix3.getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_float*CORE.getThreadCount(), CLArray.ofFloat(CORE.getThreadCount()).getPointer());
		CL.clSetKernelArg(kernel, 4, Sizeof.cl_float*CORE.getThreadCount(), CLArray.ofFloat(CORE.getThreadCount()).getPointer());
		CL.clSetKernelArg(kernel, 5, Sizeof.cl_uint, CLScalar.of(matrix1.clRows).getPointer());
		CL.clSetKernelArg(kernel, 6, Sizeof.cl_uint, CLScalar.of(matrix2.clColumns).getPointer());
		CL.clSetKernelArg(kernel, 7, Sizeof.cl_uint, CLScalar.of(matrix1.clColumns).getPointer());	
  
		for (int i = 0; i < iterations; i++)
			CORE.enqueue2DRangeKernelTest(kernel, matrix1.clRows, matrix2.clColumns, TS, TS); 
		CORE.waitOnComplete();
		long duration = System.currentTimeMillis() - start;
		double iterationsPerSecond = 1_000. / ((double)duration / iterations);
		int gflops = (int)(iterationsPerSecond * flopPerIteration / 1_000_000_000);
		System.out.printf("SGEMM took %3dms for %3d iterations and had %d GFLOPS \n", duration, iterations, gflops);  
	}
	
	protected void transpose() {
		cl_kernel kernel = kernels.get("fastTranspose").getKernel();
		    
		CL.clSetKernelArg(kernel, 0, matrix2.getSizeof(), matrix2.getPointer());
		CL.clSetKernelArg(kernel, 1, matrix2T.getSizeof(), matrix2T.getPointer());
		CL.clSetKernelArg(kernel, 2, Sizeof.cl_uint, CLScalar.of(K).getPointer());
		CL.clSetKernelArg(kernel, 3, Sizeof.cl_uint, CLScalar.of(N).getPointer());
  
		CORE.enqueue2DRangeKernelTest(kernel, matrix2.clRows, matrix2.clColumns, TRANSPOSEY, TRANSPOSEX); 
	}

	private void setUp() throws IOException, URISyntaxException {
		for (String kernelName : new String[] {"fastTranspose","oclMmul1","oclMmul2","mySGEMM1","mySGEMM2",
				"mySGEMM3","mySGEMM4","mySGEMM5","mySGEMM6","mySGEMM7","mySGEMM11","mySGEMM12"}) {			
			String sourceCode = readString(this.getClass().getResource(kernelName+".cl"));
			Subprogram<cl_kernel> subprogram = new Subprogram<>(kernelName, sourceCode, true);
			CORE.loadFromGeneratedSubprogram(subprogram);
			kernels.put(kernelName, subprogram);
		}
		CORE.compileMatrixFunctions();
		
		this.matrix1 = (CLFloatMatrix) FloatMatrix.ones(M, K, context).muli(3);
		this.matrix2 = (CLFloatMatrix) FloatMatrix.ones(K, N, context).muli(3);
		this.matrix2T = (CLFloatMatrix) FloatMatrix.ones(K, N, context).muli(3);
		this.matrix3 = (CLFloatMatrix) FloatMatrix.ones(M, N, context).muli(3);
		this.flopPerIteration = (long)M*(long)N*(2*(long)K - 1);
	}
	
	protected static String readString(URL url) throws IOException {
		String result = null;
		try(InputStream is = url.openStream()) {			
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			byte[] buffer = new byte[1024];
			int length;
			while ((length = is.read(buffer)) != -1) {
				bos.write(buffer, 0, length);
			}
			result = bos.toString("UTF-8");
		}
		return result;
	}
	
	

}
