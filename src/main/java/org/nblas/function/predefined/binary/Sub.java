package org.nblas.function.predefined.binary;

import org.nblas.function.generic.ABinaryInfix;
import org.nblas.function.generic.AFunctionObject;

public class Sub extends ABinaryInfix {
    public Sub(AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " - ";
    }
}
