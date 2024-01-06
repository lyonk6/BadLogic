# Data Parsers


## Fasta Parser
The fasta_parser is run after the uniprot parser. Its job is to fetch 
the motif sequences from a fasta file of proteins and validate modified
loci as applicable.

javac -d "bin/FastaParser"  FastaParser/*.java
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  MotifMaker
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  MotifMakerTest
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  ProteinMetaData
## UniProt Parser
The purpose of the uniprot parser is to parse for minimotifs in the 
uniprot_sprot XML data.

javac -d "bin/"  uniprot_parser/*.java

### UniProt Preprocess
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  UniProtPreprocess

### UniProt Main Parser
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  UniProtMain