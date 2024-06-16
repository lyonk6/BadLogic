package uniprot_parser;
import javax.xml.stream.*;
import javax.xml.stream.*;
import javax.xml.stream.events.*;
import javax.xml.namespace.QName;
import javax.xml.stream.events.XMLEvent;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.FileInputStream;


/**
 * The UniProtPreprocess class parses the uniprot database for evidence tags
 * that specify the evidentiary source(s) of minimotifs. 
 * 
 * Example tags:
 * 
 *  <evidence type="ECO:0000269" key="10">
 *    <source>
 *      <dbReference type="PubMed" id="24658679"/>
 *    </source>
 *  </evidence>
 *
 *  <evidence type="ECO:0000250" key="1"/>
 *  <evidence type="ECO:0000269" key="2">
 *    <source>
 *      <dbReference type="PubMed" id="32075923"/>
 *    </source>
 *  </evidence>
 *  <evidence type="ECO:0000305" key="3"/>
 * 
 */
public class UniProtPreprocess {
    public static void main(String[] args) {
        try {
            XMLInputFactory factory = XMLInputFactory.newInstance();
            XMLEventReader reader = factory.createXMLEventReader(new FileInputStream("data/may-2023/first_couple.xml"));
            //XMLEventReader reader = factory.createXMLEventReader(new FileInputStream("data/may-2023/first_100000.xml"));
            //XMLEventReader reader = factory.createXMLEventReader(new FileInputStream("data/may-2023/uniprot_sprot.xml"));

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

                    if (uniprotFeatureType.equals("evidence type")){
                        System.out.println("Does featureType ever equal \"evidence type\"?");
                    }
                    /*
                    if (uniprotFeatureType.equals("modified residue")){
                        motif.uniprotType = "modified residue";
                        motif.description=event.asStartElement().getAttributeByName(new QName("description")).toString();
                        ModifiedResidueParser.parseModifiedResidueEntries(reader, writer, motif);
                    }//*/

                   /*
                    * START_ELEMENT   1
                    * END_ELEMENT     2
                    * CHARACTERS      4
                    * ATTRIBUTE      10
                    */

                    while (reader.nextEvent().getEventType() != 2){
                        System.out.println("This is element number " + (i+1) + ": " 
                        + reader.nextEvent().getEventType());
                    }

                    /* 
                    * This is element number 1: 2
                    * This is element number 1: 4
                    * 
                    * This is element number 2: 1
                    * This is element number 2: 4
                    * 
                    * This is element number 3: 1
                    * This is element number 3: 2
                    * This is element number 3: 4
                    * 
                    * This is element number 4: 1
                    * This is element number 4: 2
                    * This is element number 4: 4
                    * 
                    * This is element number 5: 1
                    * This is element number 5: 2
                    * This is element number 5: 4
                    * 
                    * This is element number 6: 1
                    * This is element number 6: 2
                    * This is element number 6: 4
                    * 
                    */

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
