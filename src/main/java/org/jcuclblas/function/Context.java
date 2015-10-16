package org.jcuclblas.function;

/**
 * Created by Moritz on 4/27/2015.
 */
public final class  Context {

    private static final int DOUBLE = 1;
    private static final int GPU = 2;
    private static final int CUDA = 4;

    private int value;

    private Context(boolean DOUBLE, boolean GPU, boolean CUDA) {
        value |= DOUBLE ? Context.DOUBLE : 0;
        value |= GPU ? Context.GPU : 0;
        value |= CUDA ? Context.CUDA : 0;
    }

    public static Context createCudaSinglePrecisionContext() {
        return new Context(false, true, true);
    }

    public static Context createOpenCLSinglePrecisionContext() {
        return new Context(false, true, false);
    }

    public static Context createJBLASSinglePrecisionContext() {
        return new Context(false, false, false);
    }

    public static Context createCudaDoublePrecisionContext() {
        return new Context(true, true, true);
    }

    public static Context createOpenCLDoublePrecisionContext() {
        return new Context(true, true, false);
    }

    public static Context createJBLASDoublelePrecisionContext() {
        return new Context(true, false, false);
    }



    public int getValue() {
        return value;
    }

    public boolean isDouble() {
        return (Context.DOUBLE & value) == Context.DOUBLE;
    }

    public boolean isGPU() {
        return (Context.GPU & value) == Context.GPU;
    }

    public boolean isCUDA() {
        return (Context.CUDA & value) == Context.CUDA;
    }
}
