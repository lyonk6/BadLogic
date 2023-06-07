package src;
import javax.xml.stream.*;
import javax.xml.stream.events.XMLEvent;

import java.io.BufferedWriter;
import java.io.IOException;

public class PropeptideParser {
    /*
     *  <feature type="propeptide" id="PRO_0000380633" evidence="1">
     *    <location>
     *      <begin position="19"/>
     *      <end position="26"/>
     *    </location>
     *  </feature>
     */
    protected static void parsePropeptidesEntries(XMLEventReader reader, BufferedWriter writer, Minimotif motif) throws XMLStreamException {
        try{
            reader.nextEvent();  // linefeed
            XMLEvent check_tag = reader.nextEvent();  // Start "location"
            if(check_tag.isStartElement() && check_tag.asStartElement().getName().getLocalPart().equals("location")){
                reader.nextEvent(); // linefeed
                if(UniProtMain.parseLocation(reader, motif)){
                    writer.write(motif.toString() + "\n");
                }
            } else {
                System.out.println("No location found. Skipping motif: " + motif.toString());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
