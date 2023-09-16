package src;
import javax.xml.stream.*;
import javax.xml.stream.events.XMLEvent;

import java.io.BufferedWriter;
import java.io.IOException;

public class UniProtPreprocess {
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
        String accessionNumber = "";
        Minimotif motif = new Minimotif();
        try {
            writer = new BufferedWriter(new FileWriter("accession_numbers.out"));
            while (reader.hasNext()) {
                event = reader.nextEvent();/*
                if (count >= 5000000)
                    break;// */
                count++;

                if (count % 10000000 == 0)
                    System.out.println(count);

                /* Grab the accession number. Note they sometimes appear in series and we want 
                 * the first one only:
                 * uniprot_sprot.xml:150759382:  <accession>Q9P7C5</accession>
                 * uniprot_sprot.xml:150759383:  <accession>Q9UT15</accession>
                 */
                if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("accession")){
                    accessionNumber = reader.nextEvent().asCharacters().getData();
                    while(reader.hasNext()) {
                        event = reader.nextEvent();
                        if (event.isStartElement() && !event.asStartElement().getName().getLocalPart().equals("accession")){
                            break;
                        }
                    }
                }
                
                /* Look for the "feature" tags and check if it is a minimotif. 
                 */
                if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("evidence")){

                    uniprotFeatureType=event.asStartElement().getAttributeByName(new QName("type")).toString();
                    uniprotFeatureType=uniprotFeatureType.toLowerCase().strip().substring(6, uniprotFeatureType.length()-1);

                    /*
                    if (uniprotFeatureType.equals("modified residue")){
                        motif.uniprotType = "modified residue";
                        motif.description=event.asStartElement().getAttributeByName(new QName("description")).toString();
                        ModifiedResidueParser.parseModifiedResidueEntries(reader, writer, motif);
                    }//*/

                    String event_0=reader.nextEvent().asStartElement().getAttributeByName(new QName("key")).toString();
                    String event_1=reader.nextEvent().asStartElement().getAttributeByName(new QName("key")).toString();
                    String event_2=reader.nextEvent().asStartElement().getAttributeByName(new QName("key")).toString();
                    String event_3=reader.nextEvent().asStartElement().getAttributeByName(new QName("key")).toString();
                    String event_4=reader.nextEvent().asStartElement().getAttributeByName(new QName("key")).toString();
                    String event_5=reader.nextEvent().asStartElement().getAttributeByName(new QName("key")).toString();
                    String event_6=reader.nextEvent().asStartElement().getAttributeByName(new QName("key")).toString();
                    String event_7=reader.nextEvent().asStartElement().getAttributeByName(new QName("key")).toString();
                    String event_8=reader.nextEvent().asStartElement().getAttributeByName(new QName("key")).toString();
                    String event_9=reader.nextEvent().asStartElement().getAttributeByName(new QName("key")).toString();
                    EvidenceParser.parseEvidenceEntries(reader, writer);
                    //*/
                    
                    if (uniprotFeatureType.equals("evidence")){
                        parseEvidence(reader, writer, motif);
                    }//*/
                }
            }
            writer.close();
        } catch (IOException e) {
            System.out.println("Error encountered on input line: " + (count+1));
            e.printStackTrace();
        }
        return count;
    }
}
