package org.nblas.function.generic;

import org.nblas.Context;

/**
 * Created by Moritz on 4/27/2015.
 */
public abstract class AContextBasedExpression extends AFunctionObject {
    protected String operator;

    public AContextBasedExpression(AFunctionObject... children) {
        super(children);
    }
    
	protected String getFunction() {
		return operator;
	}
    
    public void setContext(Context context) {

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
