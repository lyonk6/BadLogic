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
                if (count >= 100000)
                    break;
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

                    if (featureType.equals("modified residue"))
                        System.out.println("Feature type: " + featureType);
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

                        String event_1, event_2, event_3, event_4, event_5;
                        event_1 = reader.nextEvent().asCharacters().getData(); // linefeed
                        System.out.println("Event 1: " + event_1);
                        event_2 = reader.nextEvent().asStartElement().getName().getLocalPart(); // "location" or "original"
                        if(event_2.equals("location")){
                            System.out.println("Oh boy! This is a Post-translational modification: " + event_2);
                            event_3 = reader.nextEvent().asCharacters().getData(); // linefeed
                            XMLEvent e = reader.nextEvent();
                            event_4 = e.asStartElement().getName().getLocalPart(); // start element
                            event_5 = e.asStartElement().getAttributeByName(new QName("position")).toString();// Position?
                            System.out.println(event_4 + ": " + event_5);
                        }
                        if(event_2.equals("original")){
                            System.out.println("Hey?! What the fuck is this? : " + event_2);
                            event_3 = reader.nextEvent().asCharacters().getData(); // linefeed
                            event_4 = reader.nextEvent().asEndElement().getName().getLocalPart(); // start element or end element
                            System.out.println(event_2 + " : " + event_3 + " : " + event_4);
                        }
                    }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return count;
    }
}
