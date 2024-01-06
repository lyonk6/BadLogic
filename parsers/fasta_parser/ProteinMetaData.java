package fasta_parser;
import java.util.regex.Pattern;
import java.util.regex.Matcher;  

public class ProteinMetaData {
    public String id;
    public String common_name;
    public String species;
    public String species_id;
    

    public ProteinMetaData(String s){
        String[] s_array = s.split("\\|");
        this.id = s_array[1].trim();

        //Match the common name:  FPG_DESHD
        this.common_name = parseWithPattern("\\w+", s_array[2]);

        //Match the species: OS=Rattus norvegicus 
        this.species = parseWithPattern("OS=[^\\s]+", s_array[2]);
        
        //Match the species id: OX=10116
        this.species_id = parseWithPattern("OX=\\d+", s_array[2]);
    }

    private static String parseWithPattern(String pattern, String header) {
        Pattern p = Pattern.compile(pattern);
        Matcher m = p.matcher(header);
        if (m.find()) { // Use m.find() to find a match
            return m.group(0);
        } else {
            return "unknown";
        }
    }

    public static void main(String[]args){
        String[] test_headers = {
            "sp|Q7TQM5|KPRP_RAT Keratinocyte proline-rich protein OS=Rattus norvegicus OX=10116 GN=Kprp PE=2 SV=1",
            "sp|B7MYJ7|ERA_ECO81 GTPase Era OS=Escherichia coli O81 (strain ED1a) OX=585397 GN=era PE=3 SV=1",
            "sp|O94130|CREA_BOTFU DNA-binding protein creA OS=Botryotinia fuckeliana OX=40559 GN=creA PE=3 SV=1",
            "sp|P05621|H2B2_WHEAT Histone H2B.2 OS=Triticum aestivum OX=4565 PE=1 SV=2"
        };
        ProteinMetaData pmd = null;

        for(String h: test_headers){
            pmd = new ProteinMetaData(h);
            System.out.println("ID:          " + pmd.id);
            System.out.println("Common Name: " + pmd.common_name);
            System.out.println("Species:     " + pmd.species);
            System.out.println("Species ID:  " + pmd.species_id);
        }
    }
}
