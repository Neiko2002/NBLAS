package org.nblas.generic;

/**
 * Eine Native Matrix alloziert Ressourcen die wieder frei gegeben werden müssen.
 * 
 * @author Nico
 *
 */
public abstract class ANativeMatrix extends AMatrix {
   
    protected boolean released = false;

    
    public boolean isReleased() {
        return released;
    }
    
    public abstract void free();   
}
