package org.nblas.function.generic;

/**
 * Created by Moritz on 4/19/2015.
 */
public abstract class ABinaryCommaSeparated extends AContextBasedExpression {
    public ABinaryCommaSeparated(AFunctionObject first, AFunctionObject second) {
        super(new AFunctionObject[]{first, second});
    }

    @Override
    protected String getFunction() {
        return operator + "(" + children.get(0).getFunction() + ", " + children.get(1).getFunction() + ")";
    }

}
