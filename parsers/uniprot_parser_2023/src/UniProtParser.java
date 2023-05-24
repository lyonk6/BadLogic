package src;
import javax.xml.namespace.QName;
import javax.xml.stream.*;
import javax.xml.stream.events.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;

public class UniProtParser {
    public static void main(String[] args) {
        try {
            XMLInputFactory factory = XMLInputFactory.newInstance();
            XMLEventReader reader = factory.createXMLEventReader(new FileInputStream("data/may-2023/uniprot_sprot.xml"));

            printEntryDetails(reader);

            reader.close();
        } catch (FileNotFoundException | XMLStreamException e) {
            e.printStackTrace();
        }
    }

    private static void printEntryDetails(XMLEventReader reader) throws XMLStreamException {
        int entryCount = 0;

        while (reader.hasNext()) {
            XMLEvent event = reader.nextEvent();

            if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("entry")) {
                entryCount++;

                String recommendedName = extractRecommendedName(reader);
                if (recommendedName != null) {
                    System.out.println("Recommended Name for Entry " + entryCount + ": " + recommendedName);
                    printModifiedResidues(reader);
                }
            }
        }
    }

    private static String extractRecommendedName(XMLEventReader reader) throws XMLStreamException {
        String recommendedName = null;

        while (reader.hasNext()) {
            XMLEvent event = reader.nextEvent();

            if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("recommendedName")) {
                recommendedName = extractFullName(reader);
            } else if (event.isEndElement() && event.asEndElement().getName().getLocalPart().equals("entry")) {
                // Reached the end of the entry, return the recommendedName if found
                return recommendedName;
            }
        }

        return recommendedName;
    }

    private static String extractFullName(XMLEventReader reader) throws XMLStreamException {
        String fullName = null;

        while (reader.hasNext()) {
            XMLEvent event = reader.nextEvent();

            if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("fullName")) {
                event = reader.nextEvent();
                if (event.isCharacters()) {
                    fullName = event.asCharacters().getData();
                }
            } else if (event.isEndElement() && event.asEndElement().getName().getLocalPart().equals("recommendedName")) {
                // Reached the end of the recommendedName element, return the fullName if found
                return fullName;
            }
        }

        return fullName;
    }

    private static void printModifiedResidues(XMLEventReader reader) throws XMLStreamException {
        boolean hasModifiedResidue = false;

        while (reader.hasNext()) {
            XMLEvent event = reader.nextEvent();

            if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("feature")) {
                String featureType = event.asStartElement().getAttributeByName(QName.valueOf("type")).getValue();
                if (featureType.equals("modified residue")) {
                    String modifiedResidue = extractModifiedResidue(reader);
                    if (modifiedResidue != null) {
                        hasModifiedResidue = true;
                        System.out.println("Modified Residue: " + modifiedResidue);
                    }
                }
            } else if (event.isEndElement() && event.asEndElement().getName().getLocalPart().equals("entry")) {
                // Reached the end of the entry
                if (!hasModifiedResidue) {
                    System.out.println("No Modified Residue Found");
                }
                return;
            }
        }
    }

    private static String extractModifiedResidue(XMLEventReader reader) throws XMLStreamException {
        String modifiedResidue = null;

        while (reader.hasNext()) {
            XMLEvent event = reader.nextEvent();

            if (event.isStartElement() && event.asStartElement().getName().getLocalPart().equals("original")) {
                event = reader.nextEvent();
                if (event.isCharacters()) {
                    modifiedResidue = event.asCharacters().getData();
                }
            } else if (event.isEndElement() && event.asEndElement().getName().getLocalPart().equals("feature")) {
                // Reached the end of the feature element, return the modifiedResidue if found
                return modifiedResidue;
            }
        }

        return modifiedResidue;
    }
}
