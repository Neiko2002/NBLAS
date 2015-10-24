package org.nblas.exception;

/**
 * Created by Moritz on 14.10.2015.
 */
public class AccessViolationException extends RuntimeException {

	private static final long serialVersionUID = 969875937850844219L;

	public AccessViolationException(String message) {
        super(message);
    }
}
