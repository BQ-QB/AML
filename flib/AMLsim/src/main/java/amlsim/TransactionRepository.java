package amlsim;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * AML Transaction repository (set of transactions) for performance
 * optimizations
 */
public class TransactionRepository {

    public final int size;
    private int index = 0;
    // private DecimalFormat amt_fmt;

    private int count = 0;
    private int limit = Integer.MAX_VALUE; // Number of transactions as buffer

    private long[] steps;
    private String[] descriptions;
    private double[] amounts;
    private String[] origIDs;
    private String[] destIDs;
    private String[] origBankIDs;
    private String[] destBankIDs;
    private long[] origPhoneChanges;
    private long[] destPhoneChanges;
    private long[] origDaysInBank;
    private long[] destDaysInBank;

    private float[] origBefore;
    private float[] origAfter;
    private float[] destBefore;
    private float[] destAfter;
    private boolean[] isSAR;
    private long[] alertIDs;
    private long[] modelType;

    private Map<Long, Integer> txCounter;
    private Map<Long, Integer> sarTxCounter;

    TransactionRepository(int size) {
        this.txCounter = new HashMap<>();
        this.sarTxCounter = new HashMap<>();

        this.size = size;
        this.steps = new long[size];
        this.descriptions = new String[size];
        this.amounts = new double[size];
        this.origIDs = new String[size];
        this.destIDs = new String[size];
        this.origBankIDs = new String[size];
        this.destBankIDs = new String[size];

        this.origBefore = new float[size];
        this.origAfter = new float[size];
        this.destBefore = new float[size];
        this.destAfter = new float[size];
        this.isSAR = new boolean[size];
        this.alertIDs = new long[size];
        this.modelType = new long[size];
        this.origPhoneChanges = new long[size];
        this.destPhoneChanges = new long[size];
        this.origDaysInBank = new long[size];
        this.destDaysInBank = new long[size];
    }

    void setLimit(int limit) {
        this.limit = limit;
    }

    void addTransaction(long step, String desc, double amt, String origID, String origBankID, String destID,
            String destBankID, float origBefore, float origAfter, float destBefore, float destAfter, boolean isSAR,
            long aid, long modelType, long origPhoneChange, long destPhoneChange, long origDaysInBank,
            long destDaysInBank) {

        if (origID != "-2" && amt > origBefore) {
            System.err.println("Warning: the amount of transaction is larger than the balance: " + amt + " > "
                    + origBefore + " (" + origID + " -> " + destID + ")");
        }

        if (count >= limit) {
            if (count == limit) {
                System.err.println("Warning: the number of output transactions has reached the limit: " + limit);
                flushLog();
                count++;
            }
            return;
        }

        this.steps[index] = step;
        this.descriptions[index] = desc;
        this.amounts[index] = amt;
        this.origIDs[index] = origID;
        this.destIDs[index] = destID;
        this.origBankIDs[index] = origBankID;
        this.destBankIDs[index] = destBankID;
        this.origBefore[index] = origBefore;
        this.origAfter[index] = origAfter;
        this.destBefore[index] = destBefore;
        this.destAfter[index] = destAfter;
        this.isSAR[index] = isSAR;
        this.alertIDs[index] = aid;
        this.modelType[index] = modelType;
        this.origPhoneChanges[index] = origPhoneChange;
        this.destPhoneChanges[index] = destPhoneChange;
        this.origDaysInBank[index] = origDaysInBank;
        this.destDaysInBank[index] = destDaysInBank;

        if (isSAR) {
            sarTxCounter.put(step, sarTxCounter.getOrDefault(step, 0) + 1);
        } else if (!desc.contains("CASH-")) {
            txCounter.put(step, txCounter.getOrDefault(step, 0) + 1); // Exclude cash transactions for counter
            count--;
        }

        count++;
        index++;
        if (index >= size) {
            flushLog();
        }
    }

    private double getDoublePrecision(double d) {
        // Round down amount to two digits (e.g. 12.3456 --> 12.34)
        // DecimalFormat will not be used because of its computation cost
        return (int) (d * 100) / 100.0;
    }

    void writeCounterLog(long steps, String logFile) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(logFile));
            writer.write("step,normal,SAR\n");
            for (long i = 0; i < steps; i++) {
                int numTx = txCounter.getOrDefault(i, 0);
                int numSARTx = sarTxCounter.getOrDefault(i, 0);
                writer.write(i + "," + numTx + "," + numSARTx + "\n");
            }
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    void flushLog() {
        // Flush transaction logs to the CSV file
        try {
            FileWriter writer1 = new FileWriter(new File(AMLSim.getTxLogFileName()), true);
            BufferedWriter writer = new BufferedWriter(writer1);

            for (int i = 0; i < this.index; i++) {
                writer.write(steps[i] + "," + descriptions[i] + "," + getDoublePrecision(amounts[i]) + "," + origIDs[i]
                        + "," + origBankIDs[i] + "," + origDaysInBank[i] + "," + origPhoneChanges[i] + ","
                        + getDoublePrecision(origBefore[i]) + "," + getDoublePrecision(origAfter[i]) + "," + destIDs[i]
                        + "," + destBankIDs[i] + "," + destDaysInBank[i] + "," + destPhoneChanges[i] + ","
                        + getDoublePrecision(destBefore[i])
                        + ","
                        + getDoublePrecision(destAfter[i])
                        + "," + (isSAR[i] ? "1" : "0") + "," + alertIDs[i] + "," + modelType[i] + "\n");
            }
            writer.flush();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        index = 0;
    }

}