package src;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

public class FastaParser {
    public static void main(String[] args) {
        String fastaFilePath = "data/uniprot_sprot.fasta";

        HashMap<String, String> fastaMap = parseFastaFile(fastaFilePath);

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

    public static HashMap<String, String> parseFastaFile(String filePath) {
        HashMap<String, String> fastaMap = new HashMap<>();
        try{        
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            String line;
            ProteinMetaData pmd = null;
            StringBuilder sequence = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                if (line.startsWith(">")) {
                    // Save the previous sequence (if any) and start a new one
                    if (pmd != null && sequence.length() > 0) {
                        fastaMap.put(pmd.id, sequence.toString());
                        sequence = new StringBuilder();
                    }

                    pmd = new ProteinMetaData(line);
                } else {
                    // Append the sequence line
                    sequence.append(line);
                }
            }

            // Save the last sequence in the file
            if (pmd != null && sequence.length() > 0) {
                fastaMap.put(pmd.id, sequence.toString());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return fastaMap;
    }
}
