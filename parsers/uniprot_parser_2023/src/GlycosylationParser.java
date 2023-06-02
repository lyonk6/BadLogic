package src;
import javax.xml.stream.*;
import javax.xml.stream.events.*;
import javax.xml.namespace.QName;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;


public class GlycosylationParser {

    public static void main(String[] args) {
        try {
            XMLInputFactory factory = XMLInputFactory.newInstance();
            XMLEventReader reader = factory.createXMLEventReader(new FileInputStream("data/may-2023/uniprot_sprot.xml"));

            int uniProtCount = parseEntries(reader);
            System.out.println("Number of UniProtKB objects and sub-objects: " + uniProtCount);

            reader.close();
        } catch (FileNotFoundException | XMLStreamException e) {
            e.printStackTrace();
        }
    }

    private static int parseEntries(XMLEventReader reader) throws XMLStreamException {
        BufferedWriter writer;
        XMLEvent event;
        int count = 0;
        String uniprotFeatureType;
        Minimotif motif = new Minimotif();
        try {
            writer = new BufferedWriter(new FileWriter("glycosylation_motifs.out"));
            while (reader.hasNext()) {
                event = reader.nextEvent();
                if (count >= 500000)
                    break;// */
                count++;

                if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("accession")){
                    motif.accessionNumber = reader.nextEvent().asCharacters().getData();
                }


                if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("feature")){
                    uniprotFeatureType=event.asStartElement().getAttributeByName(new QName("type")).toString();
                    uniprotFeatureType=uniprotFeatureType.toLowerCase().strip().substring(6, uniprotFeatureType.length()-1);

                    if (uniprotFeatureType.equals("glycosylation site")){
                        parseGlycosylationEntries(reader, writer, motif);
                    }
                }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return count;
    }

    private static void parseGlycosylationEntries(XMLEventReader reader, BufferedWriter writer, Minimotif motif) throws XMLStreamException {
        XMLEvent event_1, event_2, event_3, event_4, event_5, event_6, event_7;
        String motifType, motifTarget, sPosition;
        try{
            event_1 = reader.nextEvent();  // linefeed
            event_2 = reader.nextEvent();  // Start "location"
            event_3 = reader.nextEvent();  // linefeed
            event_4 = reader.nextEvent();  // Start "position"
            sPosition = event_4.asStartElement().getAttributeByName(new QName("position")).toString();
            writer.write(motif.accessionNumber + '`' + motif.motifType + '`' + motif.motifTarget + '`' + sPosition + "\n");writer.write("");
        } catch (IOException e) {
            e.printStackTrace();
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
