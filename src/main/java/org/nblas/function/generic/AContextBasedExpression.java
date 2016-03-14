package org.nblas.function.generic;

import org.nblas.Context;

/**
 * Created by Moritz on 4/27/2015.
 */
public abstract class AContextBasedExpression extends AFunctionObject {
    protected String expression;

    public AContextBasedExpression(AFunctionObject... children) {
        super(children);
    }
    
	protected String getFunction() {
		return expression;
	}
    
    public void setContext(Context context) {

        if (context.isCUDA()) {
            if (context.isDoublePrecision()) {
                expression = getDoubleCuda();
            } else {
                expression = getSingleCuda();
            }

        } else {
            if (context.isDoublePrecision()) {
                expression = getDoubleOpenCL();
            } else {
                expression = getSingleOpenCL();
            }
        }
    }

    protected abstract String getSingleCuda();

    protected abstract String getDoubleCuda();

    protected abstract String getSingleOpenCL();

    protected abstract String getDoubleOpenCL();
}
