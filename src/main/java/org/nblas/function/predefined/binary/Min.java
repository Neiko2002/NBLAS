package org.nblas.function.predefined.binary;


import org.nblas.function.generic.ABinaryCommaSeparated;
import org.nblas.function.generic.AFunctionObject;

public class Min extends ABinaryCommaSeparated {
    public Min(AFunctionObject first, AFunctionObject second) {
        super(first, second);
    }

    @Override
    protected String getSingleCuda() {
        return "fminf";
    }

    @Override
    protected String getDoubleCuda() {
        return "minf";
    }

    @Override
    protected String getSingleOpenCL() {
        return "min";
    }

    @Override
    protected String getDoubleOpenCL() {
        return "min";
    }
}
