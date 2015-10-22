package org.nblas.function.functionobjects.common;

import java.text.Format;

import org.nblas.function.functionobjects.generic.AFunctionObject;

/**
 * Created by Moritz on 4/27/2015.
 */
public class Arg extends AFunctionObject {
    private final int arg;

    public Arg(int arg) {
        super(new AFunctionObject[0]);
        this.arg = arg;
    }

    @Override
    protected String getFunction() {
        return "arg" + String.format("%02d", arg);
    }

    public int getArgNumber() {
        return arg;
    }
}
