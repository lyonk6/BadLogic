package src;
import javax.xml.stream.*;
import javax.xml.stream.events.XMLEvent;

import java.io.BufferedWriter;
import java.io.IOException;

public class PeptideParser {
    /*
     *  <feature type="peptide" id="PRO_0000000135" description="P3(40)" evidence="5">
     *    <location>
     *      <begin position="688"/>
     *      <end position="711"/>
     *    </location>
     *  </feature>
     */
    protected static void parsePeptidesEntries(XMLEventReader reader, BufferedWriter writer, Minimotif motif) throws XMLStreamException {
        try{
            reader.nextEvent();  // linefeed
            XMLEvent check_tag = reader.nextEvent();  // Start "location"
            if(check_tag.isStartElement() && check_tag.asStartElement().getName().getLocalPart().equals("location")){
                reader.nextEvent(); // linefeed
                if(UniProtMain.parseLocation(reader, motif)){
                    if(motif.description.startsWith("description")){
                        motif.description = motif.description.substring(13, motif.description.length() -1);
                    }
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
