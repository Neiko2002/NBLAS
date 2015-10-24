package org.nblas.function.predefined.binary;

import org.nblas.function.generic.ABinaryInfix;
import org.nblas.function.generic.AFunctionObject;

public class Mul extends ABinaryInfix {
    public Mul(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " * ";
    }
}
