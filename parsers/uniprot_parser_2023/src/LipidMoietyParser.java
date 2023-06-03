package src;
import javax.xml.stream.*;
import java.io.BufferedWriter;
import java.io.IOException;

public class BindingSiteParser {
    /*
     *  <feature type="binding site" evidence="12 31">
     *    <location>
     *      <begin position="135"/>
     *      <end position="136"/>
     *    </location>
     *    <ligand>
     *      <name>O-phospho-L-serine</name>
     *      <dbReference type="ChEBI" id="CHEBI:57524"/>
     *    </ligand>
     *  </feature>
     */
    protected static void parseBindingSiteEntries(XMLEventReader reader, BufferedWriter writer, Minimotif motif) throws XMLStreamException {
        motif.description=motif.description.trim().substring(13, motif.description.length()-1);
        String[] motifDescriptionArray =  motif.description.split(";");
        motif.motifType = motifDescriptionArray[0].toLowerCase().trim();
        for(String s: motifDescriptionArray){
            if (s.trim().startsWith("by ")){
                motif.motifTarget = motifDescriptionArray[1].trim().substring(3, motifDescriptionArray[1].trim().length());
                break;
            }
        }
        try{
            reader.nextEvent();  // linefeed
            reader.nextEvent();  // Start "location"
            motif.position = ModifiedResidueParser.getModifiedPosition(reader);
            writer.write(motif.accessionNumber + '`' + motif.motifType + '`' + motif.motifTarget + '`' + motif.position + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
