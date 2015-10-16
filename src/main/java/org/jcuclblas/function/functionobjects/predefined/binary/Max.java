package org.jcuclblas.function.functionobjects.predefined.binary;


import org.jcuclblas.function.functionobjects.generic.ABinaryCommaSeparated;
import org.jcuclblas.function.functionobjects.generic.AFunctionObject;

public class Max extends ABinaryCommaSeparated {
    public Max(AFunctionObject first, AFunctionObject second) {
        super(first, second);
    }

    @Override
    protected String getSingleCuda() {
        return "fmaxf";
    }

    @Override
    protected String getDoubleCuda() {
        return "maxf";
    }

    @Override
    protected String getSingleOpenCL() {
        return "max";
    }

    @Override
    protected String getDoubleOpenCL() {
        return "max";
    }
}
