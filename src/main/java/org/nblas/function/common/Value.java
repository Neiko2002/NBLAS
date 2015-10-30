package org.nblas.function.common;

import org.nblas.function.generic.AFunctionObject;
import org.nblas.function.generic.AContextBasedExpression;

public class Value extends AContextBasedExpression {
    public double value;

    public Value(double value) {
        super(new AFunctionObject[0]);
        this.value = value;
    }

    @Override
    protected String getSingleCuda() {
        return value + "f";
    }

    @Override
    protected String getDoubleCuda() {
        return value + "";
    }

    @Override
    protected String getSingleOpenCL() {
        return value + "f";
    }

    @Override
    protected String getDoubleOpenCL() {
        return value + "";
    }
}
