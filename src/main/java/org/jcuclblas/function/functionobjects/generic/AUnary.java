package org.jcuclblas.function.functionobjects.generic;

/**
 * Created by Moritz on 4/19/2015.
 */
public abstract class AUnary extends ATypedFunctionObject {

    public AUnary(AFunctionObject first) {
        super(new AFunctionObject[]{first});
    }

    @Override
    protected String getFunction() {
        return operator + "(" + children.get(0).getFunction() + ")";
    }

}
