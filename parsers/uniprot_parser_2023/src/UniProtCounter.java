package src;
import javax.xml.stream.*;
import javax.xml.stream.events.*;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;

public class UniProtCounter {
    public static void main(String[] args) {
        try {
            XMLInputFactory factory = XMLInputFactory.newInstance();
            XMLEventReader reader = factory.createXMLEventReader(new FileInputStream("data/may-2023/uniprot_sprot.xml"));

            int uniProtCount = countUniProtObjects(reader);
            System.out.println("Number of UniProtKB objects and sub-objects: " + uniProtCount);

            reader.close();
        } catch (FileNotFoundException | XMLStreamException e) {
            e.printStackTrace();
        }
    }

    private static int countUniProtObjects(XMLEventReader reader) throws XMLStreamException {
        int count = 0;
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter("qnames.out"));
            while (reader.hasNext()) {
                XMLEvent event = reader.nextEvent();
                if (count % 1000000 == 0)
                    System.out.println(count + " records processed.");
                    count++;

                if (event.isStartElement()) {
                    // Found a UniProtKB object
                    String qname = event.asStartElement().getName().getLocalPart();
                    writer.write(qname + "\n");
                    count++;
                }
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return count;
    }
}
