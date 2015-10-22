package org.nblas.function.functionobjects.predefined.unary;

import org.nblas.function.functionobjects.generic.AFunctionObject;
import org.nblas.function.functionobjects.generic.AUnary;

/**
 * Created by Moritz on 4/27/2015.
 */
public class Exp extends AUnary {
    public Exp(AFunctionObject first) {
        super(first);
    }

    @Override
    protected String getSingleCuda() {
        return "expf";
    }

    @Override
    protected String getDoubleCuda() {
        return "exp";
    }

    @Override
    protected String getSingleOpenCL() {
        return "exp";
    }

    @Override
    protected String getDoubleOpenCL() {
        return getSingleOpenCL();
    }
}
