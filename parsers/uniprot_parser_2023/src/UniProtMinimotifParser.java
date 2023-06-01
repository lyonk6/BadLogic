package src;
import javax.xml.stream.*;
import javax.xml.stream.events.*;
import javax.xml.namespace.QName;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;

public class UniProtMinimotifParser {
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
        String accessionNumber, uniprotFeatureType, motifType, motifTarget = "";
        accessionNumber = "";
        try {
            writer = new BufferedWriter(new FileWriter("accession_numbers.out"));
            while (reader.hasNext()) {
                event = reader.nextEvent();/*
                if (count >= 500000)
                    break;// */
                count++;

                if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("accession")){
                    accessionNumber = reader.nextEvent().asCharacters().getData();
                }
                    /*
                     * <feature type="modified residue" description="Phosphothreonine" evidence="3 4 9 10">
                     *   <location>
                     *     <position position="214"/>
                     *   </location>
                     * </feature>
                     */

                if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("feature")){
                    uniprotFeatureType=event.asStartElement().getAttributeByName(new QName("type")).toString();
                    uniprotFeatureType=uniprotFeatureType.toLowerCase().strip().substring(6, uniprotFeatureType.length()-1);

                    if (uniprotFeatureType.equals("modified residue")){
                        motifType=event.asStartElement().getAttributeByName(new QName("description")).toString();
                        motifType=motifType.toLowerCase().trim().substring(13, motifType.length()-1);

                        String event_1, event_2;
                        event_1 = reader.nextEvent().asCharacters().getData().trim();            // linefeed
                        event_2 = reader.nextEvent().asStartElement().getName().getLocalPart(); // "location"
                        if(event_1.equals("") && event_2.equals("location")){
                            writer.write(accessionNumber + '`' + motifType + '`' + getModifiedPosition(reader) + "\n");
                        }
                    }
                }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return count;
    }

    private static int getModifiedPosition(XMLEventReader reader) throws XMLStreamException {
        //System.out.println("Getting modified position.");
        String event_2; //, event_1;
        int position = -1;
        reader.nextEvent(); // linefeed
        XMLEvent e = reader.nextEvent();
        try{
            //event_1 = e.asStartElement().getName().getLocalPart(); // start element
            event_2 = e.asStartElement().getAttributeByName(new QName("position")).toString();
            event_2 = event_2.substring(10, event_2.length()-1);   // Position
            //System.out.println(event_1 + ": " + event_2);

            position = Integer.parseInt(event_2);
        } catch (NullPointerException npe) {
           System.err.println("This Post-translational modification is poorly formed.");
           npe.printStackTrace();
           System.exit(1);
        } catch (NumberFormatException nfe) {
            System.err.println("This position is not a number!");
            nfe.printStackTrace();
            System.exit(1);
        }
        return position;
    }
}
