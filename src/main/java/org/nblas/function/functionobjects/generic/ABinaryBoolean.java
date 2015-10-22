package org.nblas.function.functionobjects.generic;

/**
 * Created by Moritz on 4/19/2015.
 */
public abstract class ABinaryBoolean extends ABinaryEquation {
    protected String operation;

    public ABinaryBoolean(ABinaryEquation first, ABinaryEquation second) {
        super(first, second);
    }

    @Override
    protected String getFunction() {
        return children.get(0).getFunction() + operation + children.get(1).getFunction();
    }
}
