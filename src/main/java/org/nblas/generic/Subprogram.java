package org.nblas.generic;

import java.io.Reader;
import java.io.StringReader;

import net.barenca.jastyle.ASFormatter;
import net.barenca.jastyle.FormatterHelper;

/**
 * Basic Linear Algebra Subprograms
 * Jede Funktion ist ausgelagert in eine Subprogram.
 * 
 * TODO sollte seinen kernel lazy laden
 * 
 * @author Nico
 *
 */
public class Subprogram<K> {
	
	protected String programName;
	protected String sourceCode;
	protected boolean isCustom;
	
	protected K kernel;
	
	public Subprogram(String programName, String sourceCode, boolean isCustom) {
		super();
		this.programName = programName;
		this.sourceCode = sourceCode;
		this.isCustom = isCustom;
	}

	public String getProgramName() {
		return programName;
	}

	public String getSourceCode() {
		return sourceCode;
	}

	public boolean isCustom() {
		return isCustom;
	}

	public void setCustom(boolean isCustom) {
		this.isCustom = isCustom;
	}

	public K getKernel() {
		return kernel;
	}

	public void setKernel(K kernel) {
		this.kernel = kernel;
	}

	public boolean isBuild() {
		return kernel != null;
	}
	
	@Override
	public boolean equals(Object obj) {
		
		// ist es das gleiche Objekt
		if(super.equals(obj))
			return true;
		
		// ist es der gleiche Source code
		if(obj instanceof Subprogram<?>) {
			Subprogram<?> anotherSubprogram = (Subprogram<?>)obj;
			if(anotherSubprogram.getProgramName().equalsIgnoreCase(programName))
				return true;
		}
				
		return false;
	}
	
	@Override
	public String toString() {
		ASFormatter formatter = new ASFormatter();
		formatter.setSharpStyle();
		Reader reader = new StringReader(sourceCode);
		return FormatterHelper.format(reader, formatter);
	}
}
