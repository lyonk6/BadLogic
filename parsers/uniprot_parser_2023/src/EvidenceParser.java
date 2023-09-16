package src;
import javax.xml.stream.*;
import java.io.BufferedWriter;
import java.io.IOException;

public class EvidenceParser {
    protected static void parseEvidenceEntries(XMLEventReader reader, BufferedWriter writer) throws XMLStreamException {
       /*
        *   <evidence type="ECO:0000303" key="18">
        *    <source>
        *       <dbReference type="PubMed" id="22452640"/>
        *     </source>
        *   </evidence>
        */
        try {
            String event_1, event_2;
            event_1 = reader.nextEvent().asCharacters().getData().trim();            // linefeed
            event_2 = reader.nextEvent().asStartElement().getName().getLocalPart(); // "location"

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
