package src;
import javax.xml.namespace.QName;
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
            parseLocation(reader, motif);
            parseTargetLigand(reader, motif);
            writer.write(motif.toString() + "\n");
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
        reader.nextEvent(); // linefeed
        XMLEvent e1, e2, e3, e4, e5, e6, e7;
        
        e1 = reader.nextEvent(); // Start
        e2 = reader.nextEvent(); // Char
        e3 = reader.nextEvent(); // Start
        e4 = reader.nextEvent(); // Char
        e5 = reader.nextEvent(); // End
        e6 = reader.nextEvent(); // Char
        e7 = reader.nextEvent(); // Start or End
        
        // TODO validate e1 == ligand and e3 == name 
        String event_1, event_3;
        event_1 = e1.asStartElement().asStartElement().getName().getLocalPart();
        event_3 = e3.asStartElement().getName().getLocalPart();
        if(event_1.equals("ligand") && event_3.equals("name")){
            motif.motifTarget = "";
        }else{
            throw new XMLStreamException("Unexpected token while parsing target ligand.");
        }

        System.out.print("   e1.type: " + e1.getEventType());
        System.out.print("   e2.type: " + e2.getEventType());
        System.out.print("   e3.type: " + e3.getEventType());
        System.out.print("   e4.type: " + e4.getEventType());
        System.out.print("   e5.type: " + e5.getEventType());
        System.out.print("   e6.type: " + e6.getEventType());
        System.out.println("   e7.type: " + e7.getEventType());

        System.out.println("e1: " + e1.asStartElement().getName().getLocalPart());
        System.out.println("e2: " + e2.asCharacters());
        System.out.println("e3: " + e3.asStartElement().getName().getLocalPart());
        System.out.println("e4: " + e4.asCharacters());
        System.out.println("e5: " + e5.asEndElement().getName().getLocalPart());   //*/
    }

    /*
     *  <location>
     *    <begin position="135"/>
     *    <end position="136"/>
     *  </location>
     * 
     *  <location>
     *    <position position="147"/>
     *  </location>
     * 
     */
    private static void parseLocation(XMLEventReader reader, Minimotif motif) throws XMLStreamException {
        //System.out.println("Getting modified position.");
        reader.nextEvent();  // Start "location"
        reader.nextEvent(); // linefeed
        XMLEvent e1, e2, e3, e4;
        

        e1 = reader.nextEvent(); // Start: 'begin' or 'position'
        e2 = reader.nextEvent(); // End:   'begin' or 'position'
        e3 = reader.nextEvent(); // linefeed
        e4 = reader.nextEvent(); // Start or End element

        String position_type = e1.asStartElement().getName().getLocalPart();
        if (position_type == null)
            throw new XMLStreamException("Error. Motif position is null");
        try{
            switch(position_type) {
                case "begin":
                    // first parse the begin position, then parse the end position
                    String sPos, ePos;
                    sPos = e1.asStartElement().getAttributeByName(new QName("position")).toString();
                    sPos = sPos.substring(10, sPos.length()-1);   // begin position
    
                    ePos = e4.asStartElement().getAttributeByName(new QName("position")).toString();
                    ePos = ePos.substring(10, ePos.length()-1);   // begin position
    
    
                    motif.startPosition = Integer.parseInt(sPos);
                    motif.endPosition   = Integer.parseInt(ePos);
                    reader.nextEvent(); // End 
                    reader.nextEvent(); // linefeed
                    reader.nextEvent(); // End
                    break;
                case "position":
                    // e4 is the end location. 
                    String mPos;
                    mPos = e1.asStartElement().getAttributeByName(new QName("position")).toString();
                    mPos = mPos.substring(10, mPos.length()-1);   // position
                    motif.modifiedPosition = Integer.parseInt(mPos);
                    // parse the location as a singleton.
                    break;
                default:
                    throw new XMLStreamException("Unexpected token while parsing motif location: " + position_type);
            }
        } catch (NullPointerException npe) {
           System.err.println("This Post-translational modification is poorly formed.");
           npe.printStackTrace();
           System.exit(1);
        } catch (NumberFormatException nfe) {
            System.err.println("This position is not a number!");
            nfe.printStackTrace();
            System.exit(1);
        }
    }
}
