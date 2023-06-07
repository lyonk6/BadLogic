package src;
import javax.xml.stream.*;
import javax.xml.stream.events.XMLEvent;

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
     * 
     *  <feature type="binding site" evidence="7">
     *    <location>
     *      <position position="147"/>
     *    </location>
     *    <ligand>
     *      <name>Cu(2+)</name>
     *      <dbReference type="ChEBI" id="CHEBI:29036"/>
     *      <label>1</label>
     *    </ligand>
     *  </feature>
     * 
     */
    protected static void parseBindingSiteEntries(XMLEventReader reader, BufferedWriter writer, Minimotif motif) throws XMLStreamException {
        try{
            reader.nextEvent();  // linefeed
            XMLEvent check_tag = reader.nextEvent();  // Start "location"
            if(check_tag.isStartElement() && check_tag.asStartElement().getName().getLocalPart().equals("location")){
                reader.nextEvent(); // linefeed
                UniProtMain.parseLocation(reader, motif);
                parseTargetLigand(reader, motif);
                writer.write(motif.toString() + "\n");
            } else {
                System.out.println("No location found. Skipping motif: " + motif.toString());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /*
     *  <ligand>
     *    <name>O-phospho-L-serine</name>
     *    <dbReference type="ChEBI" id="CHEBI:57524"/>
     *  </ligand>
     * 
     *  <ligand>
     *    <name>Cu(2+)</name>
     *    <dbReference type="ChEBI" id="CHEBI:29036"/>
     *    <label>1</label>
     *  </ligand>
     * 
     */
    private static void parseTargetLigand(XMLEventReader reader, Minimotif motif) throws XMLStreamException {
        XMLEvent e1, e3, e4;
        
        e1 = reader.nextEvent(); // Start
        reader.nextEvent();      // Char
        e3 = reader.nextEvent(); // Start
        e4 = reader.nextEvent(); // Char
        reader.nextEvent();      // End
     
        String event_1, event_3;
        if (!e1.isStartElement()){
            System.err.println("This motif target is poorly formed. Start \"ligand\" element expected. Motif: " + motif.toString());
            return;
        }

        if (!e3.isStartElement()){
            System.err.println("This motif target is poorly formed. Start ligand \"name\" element expected. Motif: " + motif.toString());
            return;
        }

        event_1 = e1.asStartElement().asStartElement().getName().getLocalPart();
        event_3 = e3.asStartElement().getName().getLocalPart();
        if(event_1.equals("ligand") && event_3.equals("name")){
            motif.motifTarget = e4.asCharacters().toString();
        }else{
            throw new XMLStreamException("Unexpected token while parsing target ligand.");
        }
    }
}
