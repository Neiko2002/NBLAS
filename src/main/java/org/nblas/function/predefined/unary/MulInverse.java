package org.nblas.function.predefined.unary;

import org.nblas.function.generic.AFunctionObject;
import org.nblas.function.generic.AUnary;

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
