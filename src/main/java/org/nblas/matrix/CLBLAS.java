package org.nblas.matrix;

import org.jocl.*;

import java.lang.reflect.*;

/**
 * Created by Moritz on 29.08.2015.
 */
class CLBLAS {

    static {
        System.loadLibrary("joclBlas");
    }

    private final Field nativePointerField;

    public CLBLAS(long platform, long device, long context,
                  long commandQueue) throws NoSuchFieldException, IllegalAccessException {
        this.nativePointerField = NativePointerObject.class.getDeclaredField("nativePointer");
        nativePointerField.setAccessible(true);
        setup(platform,
                device,
                context,
                commandQueue);
    }

    public native void setup(long platformPointer, long devicePointer, long contextPointer, long commandQueuePointer);


    public native void sgemmNN(int M, int N, int K,
                               float alpha,
                               long bufferAPointer, long bufferBPointer,
                               float beta,
                               long bufferCPointer,
                               long eventPointer);

}
