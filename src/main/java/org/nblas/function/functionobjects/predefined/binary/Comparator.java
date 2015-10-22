package org.nblas.function.functionobjects.predefined.binary;

import org.nblas.function.functionobjects.generic.ABinaryInfix;
import org.nblas.function.functionobjects.generic.AFunctionObject;

public class Comparator extends ABinaryInfix {
    public Comparator(String comparator, AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " "+comparator+" ";
    }
}
