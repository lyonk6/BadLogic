import java.util.regex.Pattern;
import java.util.regex.Matcher;  

public class ProteinMetaData {
    public String id;
    public String common_name;
    public String species;
    public String species_id;
    

    public ProteinMetaData(String s){
        String[] s_array = s.split("|");
        this.id = s_array[1].trim();

        //Match the common name:  FPG_DESHD
        this.common_name = parseWithPattern("\\w+", s_array[2]);

        //Match the species:  OS=Desulfitobacterium hafniense 
        this.species    = parseWithPattern("OS=\\w+ \\w+", s_array[2]);

        //Match the species:  OX=272564
        this.species_id = parseWithPattern("OX=\\d+", s_array[2]);
    }

    private static String parseWithPattern(String pattern, String header){
        Pattern p = Pattern.compile(pattern);
        Matcher m = p.matcher(header);
        if (m.groupCount() > 0)
            return m.group(0);
        else
            return "unknown";
    }
}
