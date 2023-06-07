package src;
import javax.xml.stream.*;
import javax.xml.stream.events.XMLEvent;

import java.io.BufferedWriter;
import java.io.IOException;

public class TransitPeptideParser {
    /*
    *  <feature type="transit peptide" description="Chloroplast" evidence="3">
    *    <location>
    *      <begin position="1"/>
    *      <end position="51"/>
    *    </location>
    *  </feature>
    */
    protected static void parseBindingSiteEntries(XMLEventReader reader, BufferedWriter writer, Minimotif motif) throws XMLStreamException {
        try{
            reader.nextEvent();  // linefeed
            XMLEvent check_tag = reader.nextEvent();  // Start "location"
            if(check_tag.isStartElement() && check_tag.asStartElement().getName().getLocalPart().equals("location")){
                reader.nextEvent(); // linefeed
                UniProtMain.parseLocation(reader, motif);
                writer.write(motif.toString() + "\n");
            } else {
                System.out.println("No location found. Skipping motif: " + motif.toString());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
