package org.nblas.matrix;

import java.util.HashSet;

import org.nblas.function.ArgumentType;
import org.nblas.function.Context;
import org.nblas.function.functionobjects.common.Arg;
import org.nblas.function.functionobjects.common.Identity;
import org.nblas.function.functionobjects.generic.AFunctionObject;
import org.nblas.function.functionobjects.generic.ATypedFunctionObject;


abstract class AFunctionBuilder {
    private HashSet<Integer> argumentsTest;
    protected String functionName;

    public AFunctionBuilder() {
        argumentsTest = new HashSet<>();
    }

    public String buildFunction(AFunctionObject func, ArgumentType... argumentTypes) {

        argumentsTest.clear();
        AFunctionObject function = new Identity(func);
        checkFunctionArgs(function, argumentTypes.length);
        if (argumentsTest.size() != argumentTypes.length)
            throw new IllegalArgumentException("argument count or numeration is not equal to the argument types' list");
        setFunctionType(function, getContext());

        return buildFunction(function.toString(), argumentTypes);
    }

    protected abstract String buildFunction(String function, ArgumentType[] args);

    protected String generateFunctionName(String function) {
        StringBuilder builder = new StringBuilder();
        char[] chars = function.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            transform(builder, chars[i]);
        }

        return builder.toString();
    }

    public String getFunctionName() {
        return functionName;
    }

    private void transform(StringBuilder builder, char aChar) {
        if (aChar == '0') {
            builder.append('p');
        } else if (aChar == '1') {
            builder.append('q');
        } else if (aChar == '2') {
            builder.append('w');
        } else if (aChar == '3') {
            builder.append('e');
        } else if (aChar == '4') {
            builder.append('r');
        } else if (aChar == '5') {
            builder.append('t');
        } else if (aChar == '6') {
            builder.append('y');
        } else if (aChar == '7') {
            builder.append('u');
        } else if (aChar == '8') {
            builder.append('i');
        } else if (aChar == '9') {
            builder.append('o');
        } else if (aChar == '.') {
            builder.append('l');
        } else if (aChar == ',') {
            builder.append('k');
        } else if (aChar == '(') {
            builder.append('h');
        } else if (aChar == ')') {
            builder.append('j');
        } else if (aChar == '[') {
            builder.append('g');
        } else if (aChar == ']') {
            builder.append('f');
        } else if (aChar == '+') {
            builder.append('a');
        } else if (aChar == '-') {
            builder.append('s');
        } else if (aChar == '*') {
            builder.append("x");
        } else if (aChar == '/') {
            builder.append('c');
        } else if (aChar == '&') {
            builder.append('b');
        } else if (aChar == '|') {
            builder.append('n');
        } else if (aChar == '?') {
            builder.append("q_");
        } else if (aChar == ':') {
            builder.append("w_");
        } else if (aChar == '<') {
            builder.append("e_");
        } else if (aChar == '>') {
            builder.append("r_");
        } else if (aChar == '=') {
            builder.append("t_");
        } else if (aChar == '^') {
            builder.append("y_");
        } else if (aChar == ' ') {
        } else if (aChar == '%') {
            builder.append("m_");
        } else {
            builder.append(aChar);
        }
    }

    private void checkFunctionArgs(AFunctionObject function, int argsCount) {
        if (function instanceof Arg) {

            int argNumber = ((Arg) function).getArgNumber();
            if (argNumber < argsCount) {
                argumentsTest.add(argNumber);
            } else {
                throw new IllegalArgumentException("argument number is higher than the count of arguments");
            }
        }

        for (AFunctionObject functionObject : function.getChildren()) {
            checkFunctionArgs(functionObject, argsCount);
        }

    }

    public void setFunctionType(AFunctionObject function, Context context) {
        if (function instanceof ATypedFunctionObject) {
            ((ATypedFunctionObject) function).setOperator(getContext());
        }

        for (AFunctionObject functionObject : function.getChildren()) {
            setFunctionType(functionObject, getContext());
        }

    }

    protected abstract Context getContext();
}
