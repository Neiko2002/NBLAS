package org.math.joclblas;


public class Test {

    public static void main(String[] args) {

        // clTest();


//        int n = 10;
//        randomTest(n);


    }

    private static void randomTest(int n) {
        CLRandomField field = new CLRandomField(n, n);
        CLFloatMatrix matrix = CLFloatMatrix.zeros(n, n);
        field.nextGaussian(matrix, 0.0f, 1.0f);

        float[] random = matrix.toArray();
        for (int i = 0; i < random.length; i++) {
            System.out.println(random[i]);
        }
        System.out.println();
        System.out.println();
        System.out.println();
        field.nextGaussian(matrix, 0.0f, 1.0f);

        random = matrix.toArray();
        for (int i = 0; i < random.length; i++) {
            System.out.println(random[i]);
        }
    }

    private static int xorshift(int input) {
        int x = input;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        return x;
    }

    private static void clTest() {
        CLFloatMatrix.defineUnaryElementwiseFunctionX("sigmoid", "1.0f / (exp(-x) + 1.0f)");
        int i = 4096;
        int m = i;
        int n = i;
        int k = i;
        CLFloatMatrix a = CLFloatMatrix.ones(m, k);
        CLFloatMatrix b = CLFloatMatrix.ones(k, n);
        CLFloatMatrix c = CLFloatMatrix.zeros(m, n);
        long start = System.currentTimeMillis();
        CLFloatMatrix.mmul(a, b, c);

        System.out.println(System.currentTimeMillis() - start);
        float[] floats = a.toArray();

    }
}
