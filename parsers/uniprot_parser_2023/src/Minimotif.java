package src;

public class Minimotif {
    String accessionNumber, description, motifType, motifTarget;
    int position;
    
    public Minimotif(){
        this.accessionNumber = "";
        this.description = "";
        this.motifTarget = "unknown";
        this.motifType   = "";
        this.position    = -1;
    }

    public String toString(){
        return accessionNumber + '`' + motifType + '`' + motifTarget + '`' + position;
    }
}
