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
        String accessionNumber, featureType;//, featureDescription, featureEvidence;

        try {
            writer = new BufferedWriter(new FileWriter("accession_numbers.out"));
            while (reader.hasNext()) {
                event = reader.nextEvent();
                if (count >= 10000000)
                    break;// */
                count++;

                if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("accession")){
                    accessionNumber = reader.nextEvent().asCharacters().getData();
                    writer.write(accessionNumber + "\n");

                }
                    /*
                     * <feature type="modified residue" description="Phosphothreonine" evidence="3 4 9 10">
                     *   <location>
                     *     <position position="214"/>
                     *   </location>
                     * </feature>
                     */

                if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("feature")){
                    featureType=event.asStartElement().getAttributeByName(new QName("type")).toString();
                    featureType=featureType.toLowerCase().strip().substring(6, featureType.length()-1);

                    if (featureType.equals("modified residue")){
                        //System.out.println("Feature type: " + featureType);
                        //featureDescription=event.asStartElement().getAttributeByName(new QName("description")).toString();
                        //featureDescription=featureDescription.toLowerCase().strip().substring(6, featureDescription.length()-1);
    
                        // Feature evidence is null sometimes?
                        //featureEvidence=event.asStartElement().getAttributeByName(new QName("evidence")).toString();
                        //featureEvidence=featureEvidence.toLowerCase().strip().substring(6, featureEvidence.length()-1);

                        /*  These are the event types observed:
                         * 1 -> START_ELEMENT
                         * 2 -> END_ELEMENT
                         * 4 -> CHARACTERS
                         */

                        String event_1, event_2, event_3, event_4;
                        event_1 = reader.nextEvent().asCharacters().getData().trim(); // linefeed
                        System.out.println("Event 1: " + event_1);
                        event_2 = reader.nextEvent().asStartElement().getName().getLocalPart(); // "location" or "original"
                        if(event_1.equals("") && event_2.equals("location")){
                            printPTM(reader);
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

    static void printPTM(XMLEventReader reader) throws XMLStreamException {
        String event_1, event_2;
        System.out.println("Oh boy! This is a Post-translational modification!");
        reader.nextEvent(); // linefeed
        XMLEvent e = reader.nextEvent();
        try{
            event_1 = e.asStartElement().getName().getLocalPart(); // start element
            event_2 = e.asStartElement().getAttributeByName(new QName("position")).toString();// Position?
            System.out.println(event_1 + ": " + event_2);
        } catch (NullPointerException npe) {
           System.out.println("This Post-translational modification is poorly formed.");
        }
    }
}
