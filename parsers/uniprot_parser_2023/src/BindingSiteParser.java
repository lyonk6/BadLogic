package src;
import javax.xml.stream.*;
import javax.xml.stream.events.*;
import javax.xml.namespace.QName;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;

public class BindingSiteParser {
    private static void parseBindingSiteEntries(XMLEventReader reader, BufferedWriter writer, Minimotif motif) throws XMLStreamException {
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
