package org.nblas.function.predefined.binary;

import org.nblas.function.generic.ABinaryInfix;
import org.nblas.function.generic.AFunctionObject;

public class Comparator extends ABinaryInfix {
    public Comparator(String comparator, AFunctionObject... functionObjects) {
        super(functionObjects);
        operation = " "+comparator+" ";
    }
}
