package org.nblas.function.generic;

import org.nblas.function.generic.ABinaryBoolean;
import org.nblas.function.generic.AFunctionObject;


public class ATernary extends AFunctionObject {
    public ATernary(ABinaryBoolean bool, AFunctionObject isTrue, AFunctionObject isFalse) {
        super(new AFunctionObject[]{bool, isTrue, isFalse});
    }

    @Override
    public String getFunction() {
        return children.get(0).getFunction() + " ? " + children.get(1).getFunction() + ":" + children.get(2).getFunction();
    }
}
