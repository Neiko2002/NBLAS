package org.nblas.function.functionobjects.predefined.binary;

import org.nblas.function.functionobjects.generic.ABinaryInfix;
import org.nblas.function.functionobjects.generic.AFunctionObject;

public class Sub extends ABinaryInfix {
    public Sub(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " - ";
    }
}
