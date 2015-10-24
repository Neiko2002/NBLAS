package org.nblas.function.generic;

import java.util.ArrayList;

/**
 * Created by Moritz on 4/19/2015.
 */
public abstract class AFunctionObject {
    protected final ArrayList<AFunctionObject> children;

    protected AFunctionObject(AFunctionObject... children) {
        this();
        for (int i = 0; i < children.length; i++) {
            this.children.add(children[i]);
        }
    }
    
    private AFunctionObject() {
        this.children = new ArrayList<>();
    }

    protected abstract String getFunction();

   
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
