package src;

public class Minimotif {
    String accessionNumber, description, uniprotType, motifTarget;
    int modifiedPosition;
    int startPosition;
    int endPosition;
    
    public Minimotif(){
        this.accessionNumber = "";
        this.description = "";
        this.motifTarget = "unknown";
        this.uniprotType   = "";
        this.modifiedPosition = -1;
        this.startPosition    = -1;
        this.endPosition      = -1;
    }

    public String toString(){
        return accessionNumber + '`' + 
               uniprotType + '`' + 
               description + '`' + 
               motifTarget + '`' + 
               modifiedPosition + '`' + 
               startPosition + '`' + 
               endPosition;
    }
}
