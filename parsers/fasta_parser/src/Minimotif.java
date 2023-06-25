package src;

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

    public static Minimotif fromString(String s){
        Minimotif m = new Minimotif();
        try{
            s_array = s.split("`");
            m.accessionNumber  = s_array[0];
            m.sequence         = s_array[1];
            m.description      = s_array[2];
            m.motifTarget      = s_array[3];
            m.uniprotType      = s_array[4];
            m.modifiedPosition = Integer.parseInt(s_array[5]);
            m.startPosition    = Integer.parseInt(s_array[6]);
            m.endPosition      = Integer.parseInt(s_array[7]);
        } catch(NullPointerError npe){
            npe.printStackTrace()
            System.exit(1)
        } catch(NumberFormatException nfe){
            nfe.printStackTrace()
            System.exit(1)
        }
        return m;
    }
}
