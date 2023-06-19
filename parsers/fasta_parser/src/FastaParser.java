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
        for (String header : fastaMap.keySet()) {
            String sequence = fastaMap.get(header);
            System.out.println("Header: " + header);
            System.out.println("Sequence: " + sequence);
            System.out.println();
            break;
        }
    }

    public static HashMap<String, String> parseFastaFile(String filePath) {
        HashMap<String, String> fastaMap = new HashMap<>();
        try{        
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            String line;
            String header = null;
            StringBuilder sequence = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                if (line.startsWith(">")) {
                    // Save the previous sequence (if any) and start a new one
                    if (header != null && sequence.length() > 0) {
                        fastaMap.put(header, sequence.toString());
                        sequence = new StringBuilder();
                    }

                    // Extract the header (without the leading ">")
                    header = line.substring(1);
                } else {
                    // Append the sequence line
                    sequence.append(line);
                }
            }

            // Save the last sequence in the file
            if (header != null && sequence.length() > 0) {
                fastaMap.put(header, sequence.toString());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return fastaMap;
    }
}
