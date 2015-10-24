package org.nblas.function.common;

import org.nblas.function.generic.AFunctionObject;

/**
 * Created by Moritz on 4/27/2015.
 */
public class Identity extends AFunctionObject {

    public Identity(AFunctionObject child) {
        super(new AFunctionObject[]{child});
    }

    @Override
    protected String getFunction() {
        return children.toString();
    }
}
