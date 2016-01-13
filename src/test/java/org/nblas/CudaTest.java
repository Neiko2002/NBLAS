package org.nblas;

import org.jblas.FloatMatrix;
import org.nblas.cuda.CudaFloatMatrix;

import java.io.IOException;

/**
 * Created by Moritz on 4/27/2015.
 */
public class CudaTest {

    public static void main(String[] args) throws IOException {

		// Thread Count: 256 (16x16)
		// time: 210ms

		CudaFloatMatrix u = new CudaFloatMatrix(2, 2);
		u.randni();
		CudaFloatMatrix uout = new CudaFloatMatrix(2, 2);
		uout.randni();
		u.transpose(u, uout);
		System.out.println(u.toString());
		System.out.println(uout.toString2D());

		int i = 4096;
		int m = i;// 1234;
		int n = i;// 1431;
		int k = i;// 1449;

		CudaFloatMatrix a = new CudaFloatMatrix(i, i);
		a.randni();
		CudaFloatMatrix b = new CudaFloatMatrix(i, i);
		b.randni();
		CudaFloatMatrix result = new CudaFloatMatrix(i, i);
		result.randni();

		long start2 = System.currentTimeMillis();
		for (int run = 0; run < 1000; run++)
			a.sub(b, result);
		long stop2 = System.currentTimeMillis();
		System.out.println("time: " + (stop2 - start2) + "ms \n");
		
		
		
		FloatMatrix aFM = new FloatMatrix(a.toArray2());
		FloatMatrix bFM = new FloatMatrix(a.toArray2());
		FloatMatrix resultFM = new FloatMatrix(a.toArray2());
		
		boolean similar = aFM.sub(bFM).compare(resultFM, 0.0001f);
		System.out.println("similar "+ similar);
		
		
//		long start2 = System.currentTimeMillis();
//		result.mmul(a, b, result);
//		long stop2 = System.currentTimeMillis();
//		System.out.println("time: " + (stop2 - start2) + "ms \n");
//
//		long start1 = System.currentTimeMillis();
//		a.transpose(a, result);
//		result.mmul(result, b, a);
//		long stop1 = System.currentTimeMillis();
//		System.out.println("time: " + (stop1 - start1) + "ms \n");
//
//		long start = System.currentTimeMillis();
//		result.mmulTN(a, b, result);
//		long stop = System.currentTimeMillis();
//		System.out.println("time: " + (stop - start) + "ms \n");
	}


     public static void test(String[] args) throws IOException {


//        Add first = new Add(new Value(5), new Arg(0), new Arg(1));
//        AUnary functionObject = new Sine(first);
//        AFunctionBuilder builder = new CLFloatFunctionBuilder();
//        System.out.println(builder.buildFunction(functionObject, ArgumentType.MATRIIX, ArgumentType.SCALAR));
        int i = 8;
        int m = i;//1234;
        int n = i;//1431;
        int k = i;//1449;

		CudaFloatMatrix a = new CudaFloatMatrix(i, i);
        a.setOne();
        CudaFloatMatrix b = new CudaFloatMatrix(i, i);
        a.setOne();
        CudaFloatMatrix c = new CudaFloatMatrix(i,i);
        c.mmul(a, b, c);
        System.out.println(c.toString2D());

        for (int z = 0; z < 2000; z++) {
//            BufferedImage noise = new BufferedImage(i, i, BufferedImage.TYPE_INT_RGB);
//            int[] pixels = new int[i * i];
//            float[] data = new float[i * i];
//            random.getColumnWiseOn(data);
//            for (int j = 0; j < pixels.length; j++) {
//                int index = j;
//                int brightness = (int) (data[index] * 255) & 255;
//                pixels[j] = 0xFF000000 |
//                        (brightness << 16) |
//                        (brightness << 8) |
//                        (brightness << 0);
//            }
//            noise.setRGB(0, 0, i, i, pixels, 0, i);
//            ImageIO.write(noise, "png", new File("resultImages/" + z + ".png"));
//            long start = System.currentTimeMillis();
//            FloatMatrix f = FloatMatrix.randn(i, i);
//            System.out.println(System.currentTimeMillis() - start + "ms -CPU");
//            start = System.currentTimeMillis();
//            random.nextRandn();
//            System.out.println(System.currentTimeMillis() - start + "ms");
        }


//        org.jblas.FloatMatrix a = org.jblas.FloatMatrix.randn(k, m);
//        org.jblas.FloatMatrix b = org.jblas.FloatMatrix.randn(k, n);
//        org.jblas.FloatMatrix c = org.jblas.FloatMatrix.zeros(m, n);
//        CudaFloatMatrix aCL = new CudaFloatMatrix(k, m, a.data);
//        CudaFloatMatrix bCL = new CudaFloatMatrix(k, n, b.data);
//        CudaFloatMatrix cCL = new CudaFloatMatrix(m, n);
//        long start = System.currentTimeMillis();
//        CudaFloatMatrix.mmulTransposeA(aCL, bCL, cCL);
//        System.out.println((System.currentTimeMillis() - start) + "ms");
//        float[] data = new float[m * n];
//        cCL.getColumnWiseOn(data);
//        FloatMatrix d = new FloatMatrix(m, n, data);

//        start = System.currentTimeMillis();
//        a.transpose().mmuli(b, c);
//        System.out.println((System.currentTimeMillis() - start) + "ms");
//
//        System.out.println(MatrixFunctions.abs(c.sub(d)).sum() / c.length);
//        System.out.println(cCL);
//        System.out.println(c);


//        float[] values = new float[rows * columns];
//        for (int i = 0; i < values.length; i++) {
//            values[i] = i;
//        }
//
//        CudaFloatMatrix a = new CudaFloatMatrix(rows, columns, values);
//        CudaFloatMatrix b = new CudaFloatMatrix(rows, columns, values);
//        CudaFloatMatrix c = new CudaFloatMatrix(rows, columns, new float[rows * columns]);
//        CudaFloatMatrix.add(a, b, c);
//        System.out.println(a.toString2());
//        CudaFloatMatrix.add(a, b, c);
//        System.out.println(c.toString2());

//        int columns = 16;
//        int rows = 16;
//        float[] values = new float[rows * columns];
//        for (int i = 0; i < values.length; i++) {
//            values[i] = i;
//        }
        int rows = 1;
        int columns = 254;
        FloatMatrix test = FloatMatrix.rand(rows, columns);
        CudaFloatMatrix gpuTest = new CudaFloatMatrix(rows, columns, test.data);
//        CudaFloatMatrix gpuTestResult = CudaFloatMatrix.testsum(gpuTest);
//        System.out.println(gpuTestResult.toString2());
        System.out.println(columns * rows);

//        System.out.println(gpuVector);
        System.out.println(test.sum());
        System.out.println(gpuTest.sum(gpuTest));
        System.out.println(test.max());
        System.out.println(gpuTest.max(gpuTest));
        System.out.println(test.min());
        System.out.println(gpuTest.min(gpuTest));

    }
}
