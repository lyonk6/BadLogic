package src;

public class Minimotif {
    String accessionNumber, description, motifType, motifTarget;
    int modifiedPosition;
    
    public Minimotif(){
        this.accessionNumber = "";
        this.description = "";
        this.motifTarget = "unknown";
        this.motifType   = "";
        this.modifiedPosition    = -1;
    }

    public String toString(){
        return accessionNumber + '`' + motifType + '`' + motifTarget + '`' + modifiedPosition;
    }
}
