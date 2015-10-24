package org.nblas;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.nblas.cl.CLFloatMatrix;
import org.nblas.cuda.CudaFloatMatrix;
import org.nblas.function.ArgumentType;
import org.nblas.function.common.Arg;
import org.nblas.function.common.Value;
import org.nblas.function.generic.AUnary;
import org.nblas.function.predefined.binary.Add;
import org.nblas.function.predefined.unary.Sine;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Moritz on 4/27/2015.
 */
public class Test {
    public static void main(String[] args) throws IOException {
    		 
 		 	// Thread Count: 	256 (16x16)
 		 	// time: 210ms 
 		 
 		 CLFloatMatrix u = CLFloatMatrix.randn(2, 2);
 		 CLFloatMatrix uout = CLFloatMatrix.randn(2, 2);
 	     CLFloatMatrix.transpose(u, uout);
 	     System.out.println(u.toString());
 	     System.out.println(uout.toString2());
 	     
 	     
 		 
 		    int i = 4096;
 	        int m = i;//1234;
 	        int n = i;//1431;
 	        int k = i;//1449;
 	        
 	        CLFloatMatrix a = CLFloatMatrix.randn(i, i);
 	        CLFloatMatrix b = CLFloatMatrix.randn(i, i);
 	        CLFloatMatrix result = CLFloatMatrix.randn(i, i);
 	        
 	        long start2 = System.currentTimeMillis();
 	        CLFloatMatrix.mmul(a, b, result);
 	        long stop2 = System.currentTimeMillis();
 	        System.out.println("time: "+(stop2-start2)+"ms \n");
 	        
 	        long start1 = System.currentTimeMillis();
 	        CLFloatMatrix.transpose(a, result);
 	        CLFloatMatrix.mmul(result, b, a);
 	        long stop1 = System.currentTimeMillis();
 	        System.out.println("time: "+(stop1-start1)+"ms \n");
 	        
 	        long start = System.currentTimeMillis();
 	        CLFloatMatrix.mmulTransposeA(a, b, result);
 	        long stop = System.currentTimeMillis();
 	        System.out.println("time: "+(stop-start)+"ms \n");
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

        CudaFloatMatrix a = CudaFloatMatrix.ones(i, i);
        CudaFloatMatrix b = CudaFloatMatrix.ones(i, i);
        CudaFloatMatrix c = CudaFloatMatrix.zeros(i,i);
        CudaFloatMatrix.mmul(a,b, c);
        System.out.println(c.toString2());

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
//        CLFloatMatrix aCL = new CLFloatMatrix(k, m, a.data);
//        CLFloatMatrix bCL = new CLFloatMatrix(k, n, b.data);
//        CLFloatMatrix cCL = new CLFloatMatrix(m, n);
//        long start = System.currentTimeMillis();
//        CLFloatMatrix.mmulTransposeA(aCL, bCL, cCL);
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
//        CLFloatMatrix a = new CLFloatMatrix(rows, columns, values);
//        CLFloatMatrix b = new CLFloatMatrix(rows, columns, values);
//        CLFloatMatrix c = new CLFloatMatrix(rows, columns, new float[rows * columns]);
//        CLFloatMatrix.add(a, b, c);
//        System.out.println(a.toString2());
//        CLFloatMatrix.add(a, b, c);
//        System.out.println(c.toString2());

//        int columns = 16;
//        int rows = 16;
//        float[] values = new float[rows * columns];
//        for (int i = 0; i < values.length; i++) {
//            values[i] = i;
//        }
        int rows = 1;
        int columns = 254;
        org.jblas.FloatMatrix test = org.jblas.FloatMatrix.rand(rows, columns);
        CLFloatMatrix gpuTest = new CLFloatMatrix(rows, columns, test.data);
//        CLFloatMatrix gpuTestResult = CLFloatMatrix.testsum(gpuTest);
//        System.out.println(gpuTestResult.toString2());
        System.out.println(columns * rows);

//        System.out.println(gpuVector);
        System.out.println(test.sum());
        System.out.println(CLFloatMatrix.sum(gpuTest));
        System.out.println(test.max());
        System.out.println(CLFloatMatrix.max(gpuTest));
        System.out.println(test.min());
        System.out.println(CLFloatMatrix.min(gpuTest));

    }
}
