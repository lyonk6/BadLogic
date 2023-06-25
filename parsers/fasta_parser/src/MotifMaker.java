package src;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

public class MotifMaker {
    public static void main(String[] args) {
        String fastaFilePath = "data/uniprot_sprot.fasta";
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
    }

    public static void motifSequenceSearcher(String file){
        try{        
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            String line;
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1)
        }
    }
}
