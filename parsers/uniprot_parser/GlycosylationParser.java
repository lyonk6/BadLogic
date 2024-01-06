package uniprot_parser;
import javax.xml.stream.*;
import java.io.BufferedWriter;
import java.io.IOException;

public class GlycosylationParser {
    /*
     * <feature type="glycosylation site" description="N-linked (GlcNAc...) asparagine; by host" evidence="2">
     *   <location>
     *     <position position="64"/>
     *   </location>
     * </feature>
     * 
     */
    protected static void parseGlycosylationEntries(XMLEventReader reader, BufferedWriter writer, Minimotif motif) throws XMLStreamException {
        motif.description=motif.description.trim().substring(13, motif.description.length()-1);
        String[] motifDescriptionArray =  motif.description.split(";");
        motif.description = motifDescriptionArray[0].toLowerCase().trim();
        for(String s: motifDescriptionArray){
            if (s.trim().startsWith("by ")){
                motif.motifTarget = motifDescriptionArray[1].trim().substring(3, motifDescriptionArray[1].trim().length());
                break;
            }
        }
        try{
            reader.nextEvent();  // linefeed
            reader.nextEvent();  // Start "location"
            motif.modifiedPosition = UniProtMain.getModifiedPosition(reader);
            writer.write(motif.toString() + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
