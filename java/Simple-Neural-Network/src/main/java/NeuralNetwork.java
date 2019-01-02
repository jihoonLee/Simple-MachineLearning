import java.util.Arrays;
import java.util.Random;

public class NeuralNetwork {
    private int inputNum;
    private int outputNum;
    private int hiddenNum;
    private double hiddenWeight[][];
    private double outputWeight[][];
    private double inputNeuron[];
    private double hiddenNeuron[];
    private double outputNeuron[];
    private double learningLate;
    private int epoch;
    private double label[][];
    private double data[][];
    private double target[];
    private int total;
    private Random rand;
    public NeuralNetwork(int inputNum, int hiddenNum, int outputNum, double learningLate, int epoch) {
        this.inputNum = inputNum;
        this.hiddenNum = hiddenNum;
        this.outputNum = outputNum;
        this.learningLate = learningLate;
        this.epoch = epoch;
        this.rand = new Random();
        initWeight();
        initNeuron();
    }
    public void setInput(double data[][], double label[][]) {
        if(data.length != label.length) {
            System.err.println("Wrong Value");
            System.exit(1);
        }
        this.data = data;
        this.label = label;
        this.total = data.length;
    }

    private void initNeuron() {
        inputNeuron = new double[inputNum];
        hiddenNeuron = new double[hiddenNum];
        outputNeuron = new double[outputNum];
    }

    private void initWeight() {
        outputWeight = new double[outputNum][hiddenNum];
        hiddenWeight = new double[hiddenNum][inputNum];


        for(int i = 0;i < hiddenNum;i++){
            for(int j = 0;j < inputNum  ;j++){
                hiddenWeight[i][j] = rand.nextDouble() - 0.5 ;
            }
        }
        for(int i = 0;i < outputNum;i++){
            for(int j = 0;j < hiddenNum ;j++){
                outputWeight[i][j] = rand.nextDouble() - 0.5 ;
            }
        }
    }

    public void train() {
        for(int i = 0 ; i< epoch; i++) {
            for (int j = 0; j < total; j++) {
                int ch = rand.nextInt(total);
                this.target = label[ch];
                this.inputNeuron = data[ch];
                feedforward();
                backpropagation();
            }
            System.err.println(i+1 +" epoch end");
        }
    }


    public void test() {
        int count = 0;
        for(int i = 0; i<total; i++) {
            this.target = label[i];
            this.inputNeuron = data[i];
            feedforward();
            if(argmax(outputNeuron) == argmax(target))
                count++;
        }
        System.err.println("Accuracy : " + (double) count / total);
    }
    private int argmax(double[] arg) {
        int index = Integer.MIN_VALUE;
        double value = Double.MIN_VALUE;
        for(int i = 0 ; i<arg.length; i++) {
            if (value < arg[i]) {
                index = i;
                value = arg[i];
            }
        }
        return index;
    }
    private void viewResult() {
        System.out.println(Arrays.toString(outputNeuron));
    }

    private void forward(double[][] weight, double[] prev, double[] next) {
        for(int i = 0 ; i<next.length; i++) {
            double sum = 0;
            for(int j = 0 ; j< prev.length; j++) {
                sum += prev[j] * weight[i][j];
            }
            next[i]= Activation.sigmoid(sum);
        }
    }
    private void feedforward() {
        forward(hiddenWeight, inputNeuron, hiddenNeuron);
        forward(outputWeight, hiddenNeuron, outputNeuron);
    }

    private void backpropagation() {
        double[] errHidden = new double[hiddenNum];
        double[] errOutput = new double[outputNum];
        for (int i = 0; i < outputNum; i++) {
            errOutput[i] = (target[i] - outputNeuron[i] ) * (outputNeuron[i]) * (1 - outputNeuron[i]);
        }

        for (int i = 0; i < hiddenNum; i++) {
            double sum = 0;
            for (int j = 0; j < outputNum; j++) {
                sum += errOutput[j] * outputWeight[j][i];
            }
            errHidden[i] = (hiddenNeuron[i]) * (1 - hiddenNeuron[i]) * sum;
        }

        for (int i = 0; i < outputNeuron.length; i++) {
            for (int j = 0; j < hiddenNeuron.length; j++) {
                outputWeight[i][j] += learningLate * errOutput[i] * hiddenNeuron[j];
            }
        }

        for (int i = 0; i < hiddenNeuron.length; i++) {
            for (int j = 0; j < inputNeuron.length; j++) {
                hiddenWeight[i][j] += learningLate * errHidden[i] * inputNeuron[j];
            }
        }
    }

    private double getError() {
        double error = 0;
        for (int j = 0; j < outputNum; j++) {
            error += 0.5 * (outputNeuron[j] - target[j]) * (outputNeuron[j] - target[j]);
        }
        return error;
    }
}
