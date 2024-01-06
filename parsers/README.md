# Data Parsers


## Fasta Parser
The fasta_parser is run after the uniprot parser. Its job is to fetch 
the motif sequences from a fasta file of proteins and validate modified
loci as applicable.

javac -d "bin/"  src/*
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  MotifMaker

## UniProt Parser
The purpose of the uniprot parser is to parse for minimotifs in the 
uniprot_sprot XML data.


### UniProt Preprocess
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  UniProtPreprocess
java -Xms3g -Xmx3g -XX:+UseG1GC -cp "bin/"  UniProtMain