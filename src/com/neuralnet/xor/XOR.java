package com.neuralnet.xor;

import com.neuralnet.framework.NeuralNet;

import java.io.FileWriter;
import java.util.Scanner;

public class XOR {
    /**
     * This is the main program entry point that performs the following:
     * - Prompt user input on neural net parameters
     * - Create the neural net
     * - For each trial
     * -   Initialize the neural net
     * -   For each epoch, train all samples in training set
     * -     Accumulate total error and write to file
     * -     If total error < threshold then write weights data to file (optional)
     * -     Else repeat training
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        double totalError;
        double acceptError = 0.05;
        int epoch;
        FileWriter errorFile = null, weightFile = null;

        /**
         * Training data sets (inputs and outputs) for XOR including both binary and bipolar representation.
         */
        double binaryXORIn[][] = {{0,0}, {0,1}, {1,0}, {1,1}};
        double binaryXOROut[] = {0, 1, 1, 0};
        double bipolarXORIn[][] = {{-1,-1}, {-1,1}, {1,-1}, {1,1}};
        double bipolarXOROut[] = {-1, 1, 1, -1};

        double [][] trainInput;
        double [] trainOutput;

        // Neural net training parameters from user input
        NeuralNet.ActFnType actFn;
        double learningRate, momentumTerm;
        boolean saveWeight; // Y = write trained weights to output file
        int numTrial; // One trial = one complete training cycle to convergence = produce one output file

        // Prompt user input on training parameters
        Scanner userInput = new Scanner(System.in);
        System.out.print("Enter 1 for Binary, 2 for Bipolar: ");
        if (userInput.nextInt() == 1) {
            actFn = NeuralNet.ActFnType.BINARY;
        } else {
            actFn = NeuralNet.ActFnType.BIPOLAR;
        }

        System.out.print("Enter Learning Rate: ");
        learningRate = userInput.nextDouble();

        System.out.print("Enter Momentum: ");
        momentumTerm = userInput.nextDouble();

        System.out.print("Enter number of trials: ");
        numTrial = userInput.nextInt();

        if (numTrial < 1 || numTrial > 100) {
            System.out.println("Number of trials must be between 1 and 100");
            System.exit(-1);
        }

        System.out.print("Save weights to file (Y/N)? ");
        saveWeight = userInput.next().toUpperCase().toCharArray()[0] == 'Y';

        // Set up training data set depending on activation type
        if (actFn == NeuralNet.ActFnType.BINARY) {
            trainInput = binaryXORIn.clone();
            trainOutput = binaryXOROut.clone();
        } else {
            trainInput = bipolarXORIn.clone();
            trainOutput = bipolarXOROut.clone();
        }

        // Create and initialize NN
        NeuralNet xorNN = new NeuralNet(actFn,2,4, learningRate, momentumTerm, -1, 1);

        for (int t = 0; t < numTrial; t++) {
            // Initialize weights and epoch number for each trial
            xorNN.initializeWeights();
            xorNN.zeroWeights();
            epoch = 0;

            // Create output file containing epoch number and total error
            int fileSuf = t + 1;
            errorFile = xorNN.createFile("xor_out_" + fileSuf + ".txt");
            xorNN.writeHeader(errorFile);

            // Repeat training by presenting all training data in each epoch in the NN
            // Until total error is less than threshold value.
            do {
                totalError = 0;
                epoch++;

                for (int i = 0; i < trainInput.length; i++) {
                    xorNN.train(trainInput[i], trainOutput[i]);
                    totalError += xorNN.meanSqError(trainOutput[i], xorNN.activatedOutput);
                }

                // Write total error to file after each epoch
                xorNN.writeDetail(errorFile, epoch, totalError);
            } while (totalError > acceptError);

            xorNN.closeFile(errorFile);

            // Save weights to weight file if needed
            if (saveWeight) {
                weightFile = xorNN.createFile("wgt_out_" + fileSuf + ".txt");
                xorNN.save(weightFile);
                xorNN.closeFile(weightFile);
            }
        }
    }
}
