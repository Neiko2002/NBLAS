package org.nblas.function.functionobjects.generic;

/**
 * Created by Moritz on 4/19/2015.
 */
public abstract class ABinaryEquation extends AFunctionObject {
    protected String operation;

    public ABinaryEquation(AFunctionObject first, AFunctionObject second) {
        super(new AFunctionObject[]{first, second});
    }

    @Override
    protected String getFunction() {
        return children.get(0).getFunction() + operation + children.get(1).getFunction();
    }
}
