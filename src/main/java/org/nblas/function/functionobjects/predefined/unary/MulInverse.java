package org.nblas.function.functionobjects.predefined.unary;

import org.nblas.function.functionobjects.generic.AFunctionObject;
import org.nblas.function.functionobjects.generic.AUnary;

/**
 * Created by Moritz on 4/27/2015.
 */
public class MulInverse extends AUnary {
    public MulInverse(AFunctionObject first) {
        super(first);
    }

    @Override
    protected String getSingleCuda() {
        return "1.0f /";
    }

    @Override
    protected String getDoubleCuda() {
        return "1.0 /";
    }

    @Override
    protected String getSingleOpenCL() {
        return getSingleCuda();
    }

    @Override
    protected String getDoubleOpenCL() {
        return getDoubleCuda();
    }
}
