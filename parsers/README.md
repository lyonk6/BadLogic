# Data Parsers

## UniProt Parser
The purpose of the uniprot parser is to parse for minimotifs in the 
uniprot_sprot XML data and create the following table:


| Uniprot id  | uniprotType          | description      | motifTarget        | modifiedPosition   | startPosition | endPosition | 
| ----------- | -------------------- | ---------------- | -----------------  | ------------------ | ------------- | ----------- |
| E2RU97      |binding site          | general          | O-phospho-L-serine | -1                 | 135           | 136         |
| E2RU97      |modified residue      | phosphothreonine | unknown            | 214                | -1            | -1          |
| ...         | ...                  | ...              | ...                | ...                | ...           | ...         |


### Running the parser:
javac -d "bin/"  uniprot_parser/*.java

#### UniProt Preprocess
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  uniprot_parser.UniProtPreprocess

#### UniProt Main Parser
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  uniprot_parser.UniProtMain


#### Sample output:
```
E2RU97`binding site`general`O-phospho-L-serine`-1`135`136
E2RU97`modified residue`phosphothreonine`unknown`214`-1`-1
```

## Fasta Parser
The fasta_parser is run after the uniprot parser. Its job is to fetch 
the motif sequences from a fasta file of proteins and validate modified
loci as applicable.

javac -d "bin/"  fasta_parser/*.java
java -cp "bin/"                             fasta_parser.MotifMakerTest
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  fasta_parser.MotifMaker
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  fasta_parser.ProteinMetaData

## Evidence Parser
To understand the experimental orgin of entries, it is necessary to parse
their source entries. The evidence parser looks for "evidence" tags and
assembles a table that links motifs to their publication source.
