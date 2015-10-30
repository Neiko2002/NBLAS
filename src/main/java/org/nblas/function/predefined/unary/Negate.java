package org.nblas.function.predefined.unary;

import org.nblas.function.generic.AFunctionObject;
import org.nblas.function.generic.AContextBasedUnaryPrefix;

/**
 * Created by Nico
 */
public class Negate extends AContextBasedUnaryPrefix {
    public Negate(AFunctionObject first) {
        super(first);
    }

    @Override
    protected String getSingleCuda() {
        return "-";
    }

    @Override
    protected String getDoubleCuda() {
        return getSingleCuda();
    }

    @Override
    protected String getSingleOpenCL() {
        return "-";
    }

    @Override
    protected String getDoubleOpenCL() {
        return getSingleOpenCL();
    }
}
