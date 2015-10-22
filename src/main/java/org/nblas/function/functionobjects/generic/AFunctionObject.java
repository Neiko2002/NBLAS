package org.nblas.function.functionobjects.generic;

import java.util.ArrayList;

/**
 * Created by Moritz on 4/19/2015.
 */
public abstract class AFunctionObject {
    protected final ArrayList<AFunctionObject> children;

    protected AFunctionObject(AFunctionObject... children) {
        this.children = new ArrayList<>();
        for (int i = 0; i < children.length; i++) {
            this.children.add(children[i]);
        }
    }

    protected abstract String getFunction();

    private AFunctionObject() {
        children = new ArrayList<>();
    }

    @Override
    public String toString() {
        if (children.size() > 0) {
            return children.get(0).getFunction();
        } else {
            return "";
        }
    }

    public ArrayList<AFunctionObject> getChildren() {
        return children;
    }
}
