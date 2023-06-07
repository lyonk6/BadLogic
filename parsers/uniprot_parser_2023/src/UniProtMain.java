package src;
import javax.xml.stream.*;
import javax.xml.stream.events.*;
import javax.xml.namespace.QName;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;


public class UniProtMain {
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
                event = reader.nextEvent();
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
                if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("feature")){
                    motif=null;  // This is a new minimotif!!
                    motif = new Minimotif();
                    motif.accessionNumber = accessionNumber;

                    uniprotFeatureType=event.asStartElement().getAttributeByName(new QName("type")).toString();
                    uniprotFeatureType=uniprotFeatureType.toLowerCase().strip().substring(6, uniprotFeatureType.length()-1);

                    if (uniprotFeatureType.equals("modified residue")){
                        motif.uniprotType = "modified residue";
                        motif.description=event.asStartElement().getAttributeByName(new QName("description")).toString();
                        ModifiedResidueParser.parseModifiedResidueEntries(reader, writer, motif);
                    }
                    
                    if (uniprotFeatureType.equals("glycosylation site")){
                        motif.uniprotType = "glycosylation site";
                        motif.description=event.asStartElement().getAttributeByName(new QName("description")).toString();
                        GlycosylationParser.parseGlycosylationEntries(reader, writer, motif);
                    }//*/

                    if (uniprotFeatureType.equals("binding site")){
                        motif.uniprotType="binding site";
                        motif.description="general";
                        BindingSiteParser.parseBindingSiteEntries(reader, writer, motif);
                    }

                    if (uniprotFeatureType.equals("lipid moiety-binding region")){
                        motif.uniprotType="lipid moiety-binding region";
                        motif.description=event.asStartElement().getAttributeByName(new QName("description")).toString();
                        LipidMoietyParser.parseLipidMoietyEntries(reader, writer, motif);
                    }//*/
                }
            }
            writer.close();
        } catch (IOException e) {
            System.err.println("Error encountered on input line: " + (count+1));
            e.printStackTrace();
        }
        return count;
    }

    protected static int getModifiedPosition(XMLEventReader reader) throws XMLStreamException {
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
    protected static void parseLocation(XMLEventReader reader, Minimotif motif) throws XMLStreamException {
        //System.out.println("Getting modified position.");
        XMLEvent e1, e4;


        e1 = reader.nextEvent(); // Start: 'begin' or 'position'
        reader.nextEvent();      // End:   'begin' or 'position'
        reader.nextEvent();      // linefeed
        e4 = reader.nextEvent(); // Start or End element

        String position_type = e1.asStartElement().getName().getLocalPart();
        if (position_type == null)
            throw new XMLStreamException("Error. Motif position is null");
        try{
            switch(position_type) {
                case "begin":
                    // first parse the begin position, then parse the end position
                    String sPos, ePos;

                    // Parse both entries before running query:
                    sPos = e1.asStartElement().getAttributeByName(new QName("position")).toString();
                    ePos = e4.asStartElement().getAttributeByName(new QName("position")).toString();

                    sPos = sPos.substring(10, sPos.length()-1);   // begin position
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

        reader.nextEvent(); // linefeed
        } catch (NullPointerException npe) {
           System.err.println("This motif location is poorly formed. Motif: " + motif.toString());
           //Do not exit, leave the minimotif incomplete.
        } catch (NumberFormatException nfe) {
            System.err.println("This position is not a number!");
            System.err.println("Motif: " + motif.toString());
            nfe.printStackTrace();
            System.exit(1);
        }
    }
}
