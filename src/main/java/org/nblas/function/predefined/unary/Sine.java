package org.nblas.function.predefined.unary;

import org.nblas.function.generic.AFunctionObject;
import org.nblas.function.generic.AContextBasedUnaryPrefix;

public class Sine extends AContextBasedUnaryPrefix {
    public Sine(AFunctionObject first) {
        super(first);
    }

    @Override
    protected String getSingleCuda() {
        return "sinf";
    }

    @Override
    protected String getDoubleCuda() {
        return "sin";
    }

    @Override
    protected String getSingleOpenCL() {
        return "sin";
    }

    @Override
    protected String getDoubleOpenCL() {
        return getSingleOpenCL();
    }
}
