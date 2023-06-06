package src;
import javax.xml.stream.*;
import java.io.BufferedWriter;
import java.io.IOException;

public class ModifiedResidueParser {
    protected static void parseModifiedResidueEntries(XMLEventReader reader, BufferedWriter writer, Minimotif motif) throws XMLStreamException {
    /*
        * <feature type="modified residue" description="Phosphothreonine" evidence="3 4 9 10">
        *   <location>
        *     <position position="214"/>
        *   </location>
        * </feature>
        */
        try {
            motif.description=motif.description.trim().substring(13, motif.description.length()-1);
            String[] motifDescriptionArray =  motif.description.split(";");
            motif.description = motifDescriptionArray[0].toLowerCase().trim();
            for(String s: motifDescriptionArray){
                if (s.trim().startsWith("by ")){
                    motif.motifTarget = motifDescriptionArray[1].trim().substring(3, motifDescriptionArray[1].trim().length());
                    break;
                }
            }
            String event_1, event_2;
            event_1 = reader.nextEvent().asCharacters().getData().trim();            // linefeed
            event_2 = reader.nextEvent().asStartElement().getName().getLocalPart(); // "location"
            motif.modifiedPosition = UniProtMain.getModifiedPosition(reader);
            if(event_1.equals("") && event_2.equals("location")){
                writer.write(motif.toString() + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
