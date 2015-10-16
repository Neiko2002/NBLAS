package org.jcuclblas.function.functionobjects.common;

import org.jcuclblas.function.functionobjects.generic.AFunctionObject;
import org.jcuclblas.function.functionobjects.generic.ATypedFunctionObject;

public class Value extends ATypedFunctionObject {
    protected String operator;
    public double value;

    public Value(double value) {
        super(new AFunctionObject[0]);
        this.value = value;
        this.operator = getSingleOpenCL();
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

    @Override
    protected String getFunction() {
        return operator;
    }
}
