package org.jcuclblas.function.functionobjects.predefined.unary;

import org.jcuclblas.function.functionobjects.generic.AFunctionObject;
import org.jcuclblas.function.functionobjects.generic.AUnary;

public class Sine extends AUnary {
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
