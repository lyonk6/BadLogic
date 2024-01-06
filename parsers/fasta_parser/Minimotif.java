package fasta_parser;

public class Minimotif {
    String accessionNumber, description, uniprotType, motifTarget, sequence;
    int modifiedPosition;
    int startPosition;
    int endPosition;
    
    public Minimotif(){
        this.accessionNumber = "";
        this.sequence = "";
        this.description = "";
        this.motifTarget = "unknown";
        this.uniprotType   = "";
        this.modifiedPosition = -1;
        this.startPosition    = -1;
        this.endPosition      = -1;
    }

    public String toString(){
        return accessionNumber + '`' + 
               sequence + '`' +
               uniprotType + '`' + 
               description + '`' + 
               motifTarget + '`' + 
               modifiedPosition + '`' + 
               startPosition + '`' + 
               endPosition;
    }
}
