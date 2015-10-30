package org.nblas.function.generic;

/**
 * Created by Moritz on 4/19/2015.
 */
public abstract class AContextBasedUnaryPrefix extends AContextBasedExpression {

    public AContextBasedUnaryPrefix(AFunctionObject first) {
        super(first);
    }

    @Override
    protected String getFunction() {
        return operator + "(" + children.get(0).getFunction() + ")";
    }

}
