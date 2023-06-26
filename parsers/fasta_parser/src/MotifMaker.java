package src;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

public class MotifMaker {
    public static void main(String[] args) {
        String fastaFilePath = "data/uniprot_sprot.fasta";
        String motifTestPath = "sample_uniprot_motifs.txt";

        HashMap<String, String> fastaMap = FastaParser.parseFastaFile(fastaFilePath);


        // Printing the HashMap entries
        System.out.println("This is how many proteins we have " + fastaMap.size());
        for (String protein_id : fastaMap.keySet()) {
            String sequence = fastaMap.get(protein_id);
            System.out.println("Protein ID: " + protein_id);
            System.out.println("Sequence:   " + sequence);
            System.out.println();
            break;
        }

        motifSequenceSearcher(motifTestPath, fastaMap);

    }

    public static void motifSequenceSearcher(String file, HashMap<String, String> fastaMap){
        try{        
            BufferedReader reader = new BufferedReader(new FileReader(file));
            String line = "";
            Minimotif motif;
            int missingMotifs=0;
            while ((line = reader.readLine()) != null) {
                motif = fromString(line);
                if(fastaMap.get(motif.accessionNumber) != null){
                    findSequence(motif, fastaMap.get(motif.accessionNumber));
                } else {
                    System.out.println("No protein found for accession: " + motif.accessionNumber);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static Minimotif findSequence(Minimotif m, String protein){
        /*
           int modifiedPosition;
           int startPosition;
           int endPosition;
        */
       if(protein.length() <=15){
            m.sequence = "<" + protein + ">";
       }

       if(m.modifiedPosition != -1){
            if(m.modifiedPosition <= 8){
                m.sequence = "<" + protein.substring(0, m.modifiedPosition + 7);
            } else 
            if(m.modifiedPosition > protein.length()-8){
                m.sequence = protein.substring(m.modifiedPosition-8, protein.length()) + ">";
            } else {
                m.sequence = protein.substring(m.modifiedPosition -8 , m.modifiedPosition + 7);
            }
       } else {
            try{
                m.sequence = protein.substring(m.startPosition-1, m.endPosition);
                if(m.startPosition == 1){
                    m.sequence = "<" + m.sequence;
                }
                
                if(m.endPosition == protein.length()){
                    m.sequence = m.sequence + ">";
                }
            } catch (IndexOutOfBoundsException iobe){
                System.out.println("Error fetching motif. IndexOutOfBoundsException: " + m.toString());
            }
       }
       return m;
    }

    public static Minimotif fromString(String s){
        Minimotif m = new Minimotif();
        s = s.trim();
        try{
            String[] s_array = s.split("`");
            m.accessionNumber  = s_array[0];
            m.description      = s_array[1];
            m.motifTarget      = s_array[2];
            m.uniprotType      = s_array[3];
            m.modifiedPosition = Integer.parseInt(s_array[4]);
            m.startPosition    = Integer.parseInt(s_array[5]);
            m.endPosition      = Integer.parseInt(s_array[6]);
        } catch(NullPointerException npe){
            npe.printStackTrace();
            System.exit(1);
        } catch(NumberFormatException nfe){
            nfe.printStackTrace();
            System.exit(1);
        }
        return m;
    }
}
