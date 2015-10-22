package org.nblas.function.functionobjects.generic;

import org.nblas.function.Context;

/**
 * Created by Moritz on 4/27/2015.
 */
public abstract class ATypedFunctionObject extends AFunctionObject {
    protected String operator;

    public ATypedFunctionObject(AFunctionObject... children) {
        super(children);
        operator = getSingleOpenCL();
    }

    public void setOperator(Context context) {

        if (context.isCUDA()) {
            if (context.isDouble()) {
                operator = getDoubleCuda();
            } else {
                operator = getSingleCuda();
            }

        } else {
            if (context.isDouble()) {
                operator = getDoubleOpenCL();
            } else {
                operator = getSingleOpenCL();
            }
        }
    }

    protected abstract String getSingleCuda();

    protected abstract String getDoubleCuda();

    protected abstract String getSingleOpenCL();

    protected abstract String getDoubleOpenCL();
}
