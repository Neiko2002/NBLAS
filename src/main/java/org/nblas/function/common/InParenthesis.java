package org.nblas.function.common;

import org.nblas.function.generic.AFunctionObject;

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
