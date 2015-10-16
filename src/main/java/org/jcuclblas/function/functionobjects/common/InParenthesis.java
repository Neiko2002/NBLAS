package org.jcuclblas.function.functionobjects.common;

import org.jcuclblas.function.functionobjects.generic.AFunctionObject;

/**
 * Created by Moritz on 4/19/2015.
 */
public class InParenthesis extends AFunctionObject {
    public InParenthesis(AFunctionObject first) {
        super(new AFunctionObject[]{first});
    }

    @Override
    protected String getFunction() {
        return "(" + children.get(0).toString() + ")";
    }
}
