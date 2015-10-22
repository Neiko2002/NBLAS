package org.nblas.function.functionobjects.predefined.binary;

import org.nblas.function.functionobjects.generic.ABinaryInfix;
import org.nblas.function.functionobjects.generic.AFunctionObject;

public class Mul extends ABinaryInfix {
    public Mul(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " * ";
    }
}
