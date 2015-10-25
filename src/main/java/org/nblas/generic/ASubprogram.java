package org.nblas.generic;

/**
 * Basic Linear Algebra Subprograms
 * Jede Funktion ist ausgelagert in eine Subprogram.
 * 
 * @author Nico
 *
 */
public class ASubprogram {
	
	protected String programName;
	protected String sourceCode;
	protected boolean isStandardProgram;
	
	public ASubprogram(String programName, String sourceCode, boolean isStandardProgram) {
		super();
		this.programName = programName;
		this.sourceCode = sourceCode;
		this.isStandardProgram = isStandardProgram;
	}

	public String getProgramName() {
		return programName;
	}

	public String getSourceCode() {
		return sourceCode;
	}

	public boolean isStandardProgram() {
		return isStandardProgram;
	}

	public void setStandardProgram(boolean isStandardProgram) {
		this.isStandardProgram = isStandardProgram;
	}
	
}
