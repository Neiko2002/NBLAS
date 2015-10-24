package org.nblas.function.generic;

/**
 * Created by Moritz on 4/19/2015.
 */
public abstract class ABinaryInfix extends AFunctionObject {
    protected String operation;

    public ABinaryInfix(AFunctionObject... functionObjects) {
        super(functionObjects);
    }

    @Override
    protected String getFunction() {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < children.size() - 1; i++) {
            builder.append(children.get(i).getFunction() + operation);
        }
        builder.append(children.get(children.size() - 1).getFunction());
        return builder.toString();
    }
}
