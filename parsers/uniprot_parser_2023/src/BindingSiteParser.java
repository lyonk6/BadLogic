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
            getModifiedPosition(reader, motif);
            writer.write(motif.toString() + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    protected static void getModifiedPosition(XMLEventReader reader, Minimotif motif) throws XMLStreamException {
        //System.out.println("Getting modified position.");
        String event_1, event_2;
        int position = -1;
        reader.nextEvent(); // linefeed
        XMLEvent e = reader.nextEvent();
        try{
            event_1 = e.asStartElement().getAttributeByName(new QName("begin position")).toString();
            event_1 = event_1.substring(16, event_1.length()-1);   // begin position

            reader.nextEvent(); // linefeed
            event_2 = e.asStartElement().getAttributeByName(new QName("end position")).toString();
            event_2 = event_2.substring(14, event_2.length()-1);   // begin position
            position = Integer.parseInt(event_1);
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
