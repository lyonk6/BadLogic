package fasta_parser;
import java.util.HashMap;

public class MotifMakerTest {
    private static final String[] valid_motifs = {
        "Q9EUT6`N_terminal`general`sample description`-1`1`8`<MVEKRFPA",
        "Q9EUT6`N_terminal`general`sample description`-1`1`9`<MVEKRFPAA",
        "Q9EUT6`N_terminal`general`sample description`-1`1`14`<MVEKRFPAAGRDAM",
        "Q9EUT6`N_terminal`general`sample description`-1`1`15`<MVEKRFPAAGRDAMA",
        "Q9EUT6`N_terminal`general`sample description`-1`2`16`VEKRFPAAGRDAMAY",
        "Q9EUT6`N_terminal`general`sample description`1`-1`-1`<MVEKRFPA",
        "Q9EUT6`N_terminal`general`sample description`2`-1`-1`<MVEKRFPAA",
        "Q9EUT6`N_terminal`general`sample description`3`-1`-1`<MVEKRFPAAG",
        "Q9EUT6`N_terminal`general`sample description`7`-1`-1`<MVEKRFPAAGRDAM",
        "Q9EUT6`N_terminal`general`sample description`8`-1`-1`<MVEKRFPAAGRDAMA",
        "Q9EUT6`N_terminal`general`sample description`9`-1`-1`VEKRFPAAGRDAMAY",
        "C4LHU9`C_terminal`general`sample description`-1`398`411`CVPKNYELHGADED>",
        "C4LHU9`C_terminal`general`sample description`-1`397`411`VCVPKNYELHGADED>",
        "C4LHU9`C_terminal`general`sample description`-1`396`411`VVCVPKNYELHGADED>",
        "C4LHU9`C_terminal`general`sample description`-1`395`410`GVVCVPKNYELHGADE",
        "C4LHU9`C_terminal`general`sample description`-1`394`409`GGVVCVPKNYELHGAD",
        "C4LHU9`C_terminal`general`sample description`400`-1`-1`QGGVVCVPKNYELHG",
        "C4LHU9`C_terminal`general`sample description`401`-1`-1`GGVVCVPKNYELHGA",
        "C4LHU9`C_terminal`general`sample description`402`-1`-1`GVVCVPKNYELHGAD",
        "C4LHU9`C_terminal`general`sample description`403`-1`-1`VVCVPKNYELHGADE",
        "C4LHU9`C_terminal`general`sample description`404`-1`-1`VCVPKNYELHGADED>"
    };

    public static void main(String[] args) {
        HashMap<String, String> fastaMap = FastaParser.parseFastaFile("data/uniprot_sprot.fasta");
        Minimotif m = new Minimotif();
        for(String s: valid_motifs){
            String[] s_array = s.split("`");
            m = MotifMaker.fromString(s);
            m = MotifMaker.findSequence(m, fastaMap.get(m.accessionNumber));
            assertEquals(m.accessionNumber, s_array[0]);
            assertEquals(m.sequence, s_array[7]);
        }
    }

    public static void assertEquals(String one, String two){
        if (one.equals(two)){
            System.out.println(one + " : " + two);
            return;
        } else {
            String message = "Error testing MotifMaker. String mismatch: ";
            System.out.println(message + one + " : " + two);
            System.exit(1);
        }
    }
}
