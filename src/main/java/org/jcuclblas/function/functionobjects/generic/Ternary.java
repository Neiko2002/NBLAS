package org.jcuclblas.function.functionobjects.generic;

import org.jcuclblas.function.functionobjects.generic.ABinaryBoolean;
import org.jcuclblas.function.functionobjects.generic.AFunctionObject;


public class Ternary extends AFunctionObject {
    public Ternary(ABinaryBoolean bool, AFunctionObject isTrue, AFunctionObject isFalse) {
        super(new AFunctionObject[]{bool, isTrue, isFalse});
    }

    @Override
    public String getFunction() {
        return children.get(0).getFunction() + " ? " + children.get(1).getFunction() + ":" + children.get(2).getFunction();
    }
}
