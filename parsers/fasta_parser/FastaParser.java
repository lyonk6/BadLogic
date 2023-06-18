import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

public class FastaParser {
    public static void main(String[] args) {
        String fastaFilePath = "data/uniprot_sprot.fasta";

        HashMap<String, String> proteinMap = parseFastaFile(fastaFilePath);

        // Printing the HashMap entries
        for (String header : proteinMap.keySet()) {
            String sequence = proteinMap.get(header);
            System.out.println(header);
            System.out.println(sequence);
            //System.out.println();
        }
    }

    public static HashMap<String, String> parseFastaFile(String filePath) {
        String line;
        StringBuilder sequence = new StringBuilder();
        HashMap<String, String> proteinMap = new HashMap<>();
        ProteinMetaData proteinMetaData = null;
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {


            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                if (line.startsWith(">")) {
                    // Save the previous sequence (if any) and start a new one
                    proteinMetaData = new ProteinMetaData(line);

                    proteinMap.put(proteinMetaData.id, sequence.toString());
                    sequence = new StringBuilder();

                    // Extract the header (without the leading ">")
                    header = line.substring(1);
                } else {
                    // Append the sequence line
                    sequence.append(line);
                }
            }
            // Save the last sequence in the file
            proteinMap.put(proteinMetaData.id, sequence.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }

        return proteinMap;
    }
}
