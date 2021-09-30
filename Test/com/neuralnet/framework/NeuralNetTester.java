package com.neuralnet.framework;

import org.junit.Assert;
import org.junit.Test;
import java.io.FileWriter;

/** Test Driven Development (TDD) approach is used where the software is made as modular as possible via Java methods.
 * The test cases of each method are written in JUnit that drives the actual coding of the method.
 * This ensures each method is tested thoroughly in terms of expected functionality and code coverage.
 */
public class NeuralNetTester {
    // Test binary sigmoid function.  Expected values are computed in Excel.
    @Test
    public void testBinarySigmoid() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, 0, 1);

        Assert.assertEquals(0.5, testNN.binarySigmoid(0), 0.005);
        Assert.assertEquals(1.0, testNN.binarySigmoid(100), 0.005);
        Assert.assertEquals(0.0, testNN.binarySigmoid(-100), 0.005);
        Assert.assertEquals(0.88, testNN.binarySigmoid(2), 0.005);
        Assert.assertEquals(0.12, testNN.binarySigmoid(-2), 0.005);
    }

    // Test bipolar sigmoid function.  Expected values are computed in Excel.
    @Test
    public void testBipolarSigmoid() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BIPOLAR, 2, 4, 0.2, 0.9, 0, 1);

        Assert.assertEquals(0.0, testNN.bipolarSigmoid(0), 0.005);
        Assert.assertEquals(1.0, testNN.bipolarSigmoid(100), 0.005);
        Assert.assertEquals(-1.0, testNN.bipolarSigmoid(-100), 0.005);
        Assert.assertEquals(0.76, testNN.bipolarSigmoid(2), 0.005);
        Assert.assertEquals(-0.76, testNN.bipolarSigmoid(-2), 0.005);
    }

    // Test binary custom function.  Same as testBipolarSigmoid() with range -1 to 1.
    @Test
    public void testCustomSigmoid() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.CUSTOM, 2, 4, 0.2, 0.9, -1, 1);

        Assert.assertEquals(0.0, testNN.customSigmoid(0), 0.005);
        Assert.assertEquals(1.0, testNN.customSigmoid(100), 0.005);
        Assert.assertEquals(-1.0, testNN.customSigmoid(-100), 0.005);
        Assert.assertEquals(0.76, testNN.customSigmoid(2), 0.005);
        Assert.assertEquals(-0.76, testNN.customSigmoid(-2), 0.005);
    }

    // Test binary sigmoid derivative function.  Expected values are computed in Excel.
    @Test
    public void testDeriBinarySigmoid() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, 0, 1);

        Assert.assertEquals(0, testNN.deriBinarySigmoid(0), 0.005);
        Assert.assertEquals(0, testNN.deriBinarySigmoid(1), 0.005);
        Assert.assertEquals(0.25, testNN.deriBinarySigmoid(0.5), 0.005);
        Assert.assertEquals(-0.75, testNN.deriBinarySigmoid(-0.5), 0.005);
    }

    // Test bipolar sigmoid derivative function.  Expected values are computed in Excel.
    @Test
    public void testDeriBipolarSigmoid() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, 0, 1);

        Assert.assertEquals(0.5, testNN.deriBipolarSigmoid(0), 0.005);
        Assert.assertEquals(0, testNN.deriBipolarSigmoid(1), 0.005);
        Assert.assertEquals(0.375, testNN.deriBipolarSigmoid(0.5), 0.005);
        Assert.assertEquals(0.375, testNN.deriBipolarSigmoid(-0.5), 0.005);
    }

    // Test Neural Net constructor by verifying the inputs to the constructor.
    @Test
    public void testNeuralNet() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, -1, 1);

        Assert.assertEquals(4, testNN.weightsI2H.length);
        Assert.assertEquals(3, testNN.weightsI2H[0].length);
        Assert.assertEquals(4, testNN.inducedLocalHidden.length);
        Assert.assertEquals(4, testNN.activatedHidden.length);
    }

    // Test initialize weight function by verifying that the weights are between -0.5 and 0.5
    @Test
    public void testInitializeWeights() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, -1, 1);

        testNN.initializeWeights();
        Assert.assertTrue(-0.5 <= testNN.weightsH2O[0] && testNN.weightsH2O[0] <= 0.5);
        Assert.assertTrue(-0.5 <= testNN.weightsH2O[4] && testNN.weightsH2O[4] <= 0.5);
        Assert.assertTrue(-0.5 <= testNN.weightsI2H[0][0] && testNN.weightsI2H[0][0] <= 0.5);
        Assert.assertTrue(-0.5 <= testNN.weightsI2H[3][2] && testNN.weightsI2H[3][2] <= 0.5);
    }

    // Test zero weight function to verifying the weights become zero.
    @Test
    public void testZeroWeights() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, -1, 1);

        testNN.zeroWeights();

        Assert.assertEquals(0, testNN.oldWeightsI2H[0][0], 0.0005);
        Assert.assertEquals(0, testNN.oldWeightsH2O[4], 0.0005);
    }

    // Test neural net output function by loading a hardcoded set of weights.  Expected values are computed in Excel.
    @Test
    public void testOutputFor() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, -1, 1);
        NeuralNet testNN2 = new NeuralNet(NeuralNet.ActFnType.BIPOLAR, 2, 4, 0.2, 0.9, -1, 1);

        double[][] weightsI2H = {{0.1, 0.2, -0.3}, {-0.4, 0.1, -0.1}, {0.45, -0.49, 0.22}, {0.35, -0.18, 0.3}};
        double[] weightsH2O = {0.42, -0.37, 0.21, 0.39, -0.45};
        double[] binInput1 = {1, 1};
        double[] binInput2 = {1, 0};
        double[] bipolarInput1 = {1, 1};
        double[] bipolarInput2 = {1, -1};

        testNN.loadWeights(weightsI2H, weightsH2O);

        Assert.assertEquals(0.4915, testNN.outputFor(binInput1), 0.0005);
        Assert.assertEquals(0.4986, testNN.outputFor(binInput2), 0.0005);

        testNN2.loadWeights(weightsI2H, weightsH2O);

        Assert.assertEquals(-0.1333, testNN2.outputFor(bipolarInput1), 0.0005);
        Assert.assertEquals(-0.081, testNN2.outputFor(bipolarInput2), 0.0005);
    }

    // Test delta value at output layer using both binary and polar activations.  Expected values are computed in Excel.
    @Test
    public void testBpErrorOutput() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, -1, 1);
        NeuralNet testNN2 = new NeuralNet(NeuralNet.ActFnType.BIPOLAR, 2, 4, 0.2, 0.9, -1, 1);

        double[][] weightsI2H = {{0.1, 0.2, -0.3}, {-0.4, 0.1, -0.1}, {0.45, -0.49, 0.22}, {0.35, -0.18, 0.3}};
        double[] weightsH2O = {0.42, -0.37, 0.21, 0.39, -0.45};
        double[] binInput1 = {1, 1};
        double[] binInput2 = {1, 0};
        double binOutput1 = 1, binOutput2 = 0;
        double[] bipolarInput1 = {1, 1};
        double[] bipolarInput2 = {1, -1};
        double bipolarOutput1 = 1, bipolarOutput2 = -1;

        testNN.loadWeights(weightsI2H, weightsH2O);
        testNN.outputFor(binInput1);
        testNN.bpErrorOutput(binOutput1);

        Assert.assertEquals(0.1271, testNN.deltaOutput, 0.0005);

        testNN.outputFor(binInput2);
        testNN.bpErrorOutput(binOutput2);

        Assert.assertEquals(-0.1247, testNN.deltaOutput, 0.0005);

        testNN2.loadWeights(weightsI2H, weightsH2O);
        testNN2.outputFor(bipolarInput1);
        testNN2.bpErrorOutput(bipolarOutput1);

        Assert.assertEquals(0.5566, testNN2.deltaOutput, 0.0005);

        testNN2.outputFor(bipolarInput2);
        testNN2.bpErrorOutput(bipolarOutput2);

        Assert.assertEquals(-0.4565, testNN2.deltaOutput, 0.0005);

    }

    // Test delta value at hidden layer using both binary and polar activations.  Expected values are computed in Excel.
    @Test
    public void testBpErrorHidden() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, -1, 1);
        NeuralNet testNN2 = new NeuralNet(NeuralNet.ActFnType.BIPOLAR, 2, 4, 0.2, 0.9, -1, 1);

        double[][] weightsI2H = {{0.1, 0.2, -0.3}, {-0.4, 0.1, -0.1}, {0.45, -0.49, 0.22}, {0.35, -0.18, 0.3}};
        double[] weightsH2O = {0.42, -0.37, 0.21, 0.39, -0.45};
        double[] binInput1 = {1, 1};
        double[] binInput2 = {1, 0};
        double binOutput1 = 1, binOutput2 = 0;
        double[] bipolarInput1 = {1, 1};
        double[] bipolarInput2 = {1, -1};
        double bipolarOutput1 = 1, bipolarOutput2 = -1;

        testNN.loadWeights(weightsI2H, weightsH2O);
        testNN.outputFor(binInput1);
        testNN.bpErrorOutput(binOutput1);
        testNN.bpErrorHidden();

        Assert.assertEquals(0.0133, testNN.deltaHidden[0], 0.0005);
        Assert.assertEquals(-0.0113, testNN.deltaHidden[1], 0.0005);
        Assert.assertEquals(0.0067, testNN.deltaHidden[2], 0.0005);
        Assert.assertEquals(0.0117, testNN.deltaHidden[3], 0.0005);

        testNN.outputFor(binInput2);
        testNN.bpErrorOutput(binOutput2);
        testNN.bpErrorHidden();

        Assert.assertEquals(-0.0130, testNN.deltaHidden[0], 0.0005);
        Assert.assertEquals(0.0108, testNN.deltaHidden[1], 0.0005);
        Assert.assertEquals(-0.0059, testNN.deltaHidden[2], 0.0005);
        Assert.assertEquals(-0.0110, testNN.deltaHidden[3], 0.0005);

        testNN2.loadWeights(weightsI2H, weightsH2O);
        testNN2.outputFor(bipolarInput1);
        testNN2.bpErrorOutput(bipolarOutput1);
        testNN2.bpErrorHidden();

        Assert.assertEquals(0.1169, testNN2.deltaHidden[0], 0.0005);
        Assert.assertEquals(-0.0990, testNN2.deltaHidden[1], 0.0005);
        Assert.assertEquals(0.0580, testNN2.deltaHidden[2], 0.0005);
        Assert.assertEquals(0.1028, testNN2.deltaHidden[3], 0.0005);

        testNN2.outputFor(bipolarInput2);
        testNN2.bpErrorOutput(bipolarOutput2);
        testNN2.bpErrorHidden();

        Assert.assertEquals(-0.0921, testNN2.deltaHidden[0], 0.0005);
        Assert.assertEquals(0.0773, testNN2.deltaHidden[1], 0.0005);
        Assert.assertEquals(-0.0348, testNN2.deltaHidden[2], 0.0005);
        Assert.assertEquals(-0.0753, testNN2.deltaHidden[3], 0.0005);
    }

    // Test delta weights at both hidden and output layers using binary activation by hardcoded old and new weights.
    @Test
    public void testDeltaWeights() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, -1, 1);

        double[][] weightsI2H = {{0.1, 0.2, -0.3}, {-0.4, 0.1, -0.1}, {0.45, -0.49, 0.22}, {0.35, -0.18, 0.3}};
        double[] weightsH2O = {0.42, -0.37, 0.21, 0.39, -0.45};
        double[][] oldWeightsI2H = {{-0.1, -0.2, -0.3}, {0.1, 0.2, 0.3}, {0, -0.1, 0.4}, {-0.4, 0.3, 0}};
        double[] oldWeightsH2O = {-0.1, 0.2, 0.3, 0.4, 0};

        testNN.loadWeights(weightsI2H, weightsH2O);
        testNN.loadOldWeights(oldWeightsI2H, oldWeightsH2O);

        Assert.assertEquals(0.2, testNN.deltaWeightsI2H(0, 0), 0.0005);
        Assert.assertEquals(0, testNN.deltaWeightsI2H(3, 2), 0.0005);
        Assert.assertEquals(0.52, testNN.deltaWeightsH2O(0), 0.0005);
        Assert.assertEquals(0, testNN.deltaWeightsH2O(4), 0.0005);
    }

    // Test update weight function by going through the back propagation algorithm step by step using a fixed set
    // of weights and binary activation.  THe updated weights are checked against expected values computed in Excel.
    @Test
    public void testUpdateWeights() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, -1, 1);

        double[][] weightsI2H = {{0.1, 0.2, -0.3}, {-0.4, 0.1, -0.1}, {0.45, -0.49, 0.22}, {0.35, -0.18, 0.3}};
        double[] weightsH2O = {0.42, -0.37, 0.21, 0.39, -0.45};
        double[] binInput1 = {1, 1};
        double binOutput1 = 1;

        testNN.loadWeights(weightsI2H, weightsH2O);
        testNN.zeroWeights();
        testNN.outputFor(binInput1);
        testNN.bpErrorOutput(binOutput1);
        testNN.bpErrorHidden();
        testNN.updateWeightsH2O();

        Assert.assertEquals(0.4327, testNN.weightsH2O[0], 0.0005);
        Assert.assertEquals(-0.3598, testNN.weightsH2O[1], 0.0005);
        Assert.assertEquals(0.2239, testNN.weightsH2O[2], 0.0005);
        Assert.assertEquals(0.4056, testNN.weightsH2O[3], 0.0005);
        Assert.assertEquals(-0.4246, testNN.weightsH2O[4], 0.0005);

        testNN.updateWeightsI2H(binInput1);
        Assert.assertEquals(0.1027, testNN.weightsI2H[0][0], 0.0005);
        Assert.assertEquals(0.2027, testNN.weightsI2H[0][1], 0.0005);
        Assert.assertEquals(-0.2973, testNN.weightsI2H[0][2], 0.0005);
        Assert.assertEquals(-0.4023, testNN.weightsI2H[1][0], 0.0005);
        Assert.assertEquals(0.0977, testNN.weightsI2H[1][1], 0.0005);
        Assert.assertEquals(-0.1023, testNN.weightsI2H[1][2], 0.0005);
        Assert.assertEquals(0.4513, testNN.weightsI2H[2][0], 0.0005);
        Assert.assertEquals(-0.4887, testNN.weightsI2H[2][1], 0.0005);
        Assert.assertEquals(0.2213, testNN.weightsI2H[2][2], 0.0005);
        Assert.assertEquals(0.3523, testNN.weightsI2H[3][0], 0.0005);
        Assert.assertEquals(-0.1777, testNN.weightsI2H[3][1], 0.0005);
        Assert.assertEquals(0.3023, testNN.weightsI2H[3][2], 0.0005);
    }

    // Test train function by using a fixed set of weights and bipolar activation.
    // THe updated weights are checked against expected values computed in Excel.
    @Test
    public void testTrain() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BIPOLAR, 2, 4, 0.2, 0.9, -1, 1);

        double[][] weightsI2H = {{0.1, 0.2, -0.3}, {-0.4, 0.1, -0.1}, {0.45, -0.49, 0.22}, {0.35, -0.18, 0.3}};
        double[] weightsH2O = {0.42, -0.37, 0.21, 0.39, -0.45};
        double[] bipolarInput1 = {1, -1};
        double bipolarOutput1 = -1;

        testNN.loadWeights(weightsI2H, weightsH2O);
        testNN.zeroWeights();
        testNN.train(bipolarInput1, bipolarOutput1);

        Assert.assertEquals(0.4380, testNN.weightsH2O[0], 0.0005);
        Assert.assertEquals(-0.3434, testNN.weightsH2O[1], 0.0005);
        Assert.assertEquals(0.1623, testNN.weightsH2O[2], 0.0005);
        Assert.assertEquals(0.3541, testNN.weightsH2O[3], 0.0005);
        Assert.assertEquals(-0.5413, testNN.weightsH2O[4], 0.0005);

        Assert.assertEquals(0.0808, testNN.weightsI2H[0][0], 0.0005);
        Assert.assertEquals(0.2192, testNN.weightsI2H[0][1], 0.0005);
        Assert.assertEquals(-0.3192, testNN.weightsI2H[0][2], 0.0005);
        Assert.assertEquals(-0.3857, testNN.weightsI2H[1][0], 0.0005);
        Assert.assertEquals(0.0857, testNN.weightsI2H[1][1], 0.0005);
        Assert.assertEquals(-0.0857, testNN.weightsI2H[1][2], 0.0005);
        Assert.assertEquals(0.4446, testNN.weightsI2H[2][0], 0.0005);
        Assert.assertEquals(-0.4846, testNN.weightsI2H[2][1], 0.0005);
        Assert.assertEquals(0.2146, testNN.weightsI2H[2][2], 0.0005);
        Assert.assertEquals(0.3363, testNN.weightsI2H[3][0], 0.0005);
        Assert.assertEquals(-0.1663, testNN.weightsI2H[3][1], 0.0005);
        Assert.assertEquals(0.2863, testNN.weightsI2H[3][2], 0.0005);
    }

    // Test file operations including creation, writing and closing.
    // No assertion is used.  Checked by manual browsing on output file.
    @Test
    public void testFile() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BIPOLAR, 2, 4, 0.2, 0.9, -1, 1);

        FileWriter file;
        file = testNN.createFile("test_out_file.txt");
        testNN.writeHeader(file);
        testNN.writeDetail(file, 1, 1.2345);
        testNN.writeDetail(file, 2, -2.4567);
        testNN.closeFile(file);
    }

    // Test mean square error calculation.
    @Test
    public void testMeanSqError() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BIPOLAR, 2, 4, 0.2, 0.9, -1, 1);

        Assert.assertEquals(0.045, testNN.meanSqError(0.1, 0.4), 0.0005);
        Assert.assertEquals(0, testNN.meanSqError(-0.2, -0.2), 0.0005);
    }

    // Test save function where the weights are written to an output file.
    // No assertion is used.  Checked by manual browsing on output file.
    @Test
    public void testSave() {
        NeuralNet testNN = new NeuralNet(NeuralNet.ActFnType.BINARY, 2, 4, 0.2, 0.9, -1, 1);

        double[][] weightsI2H = {{0.1, 0.2, -0.3}, {-0.4, 0.1, -0.1}, {0.45, -0.49, 0.22}, {0.35, -0.18, 0.3}};
        double[] weightsH2O = {0.42, -0.37, 0.21, 0.39, -0.45};

        FileWriter file;
        file = testNN.createFile("test_weight_file.txt");

        testNN.loadWeights(weightsI2H, weightsH2O);
        testNN.save(file);
        testNN.closeFile(file);
    }
}
