import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

public class Run {
    public static void main(String[] args) {
        System.err.println("Load Data");
        DataSet trainSet = new DataSet(60000);
        readText(trainSet, "/mnistTrain.txt");

        System.err.println("Train....");
        NeuralNetwork nn = new NeuralNetwork(784, 30, 10, 0.01, 3);
        nn.setInput(trainSet.data, trainSet.label);
        nn.train();

        System.err.println("Test....");
        DataSet testSet = new DataSet(10000);
        readText(testSet, "/mnistTest.txt");
        nn.setInput(testSet.data, testSet.label);
        nn.test();

    }

    public static void readText(DataSet dataSet, String path) {
        try {
            BufferedReader in = new BufferedReader(new FileReader(Run.class.getClass().getResource(path).getPath()));
            String line;
            int count = 0;
            while ((line = in.readLine()) != null) {
                double[] data = new double[784];
                double[] label = new double[10];
                String tmp[] = line.split(" ");
                for(int i = 0; i<data.length; i++) {
                    double value = Double.valueOf(tmp[i]);
                    if(value>0) data[i] = 1;
                }
                int labelPos = Integer.valueOf(tmp[784]);
                label[labelPos] = 1;
                dataSet.data[count] = data;
                dataSet.label[count] = label;
                count++;
            }
            System.err.println("Data Size : " + count);
            in.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void viewDataSet(double[] data, double[] label) {
        System.out.println("Data Example");
        for(int i = 0; i< 28; i++) {
            for(int j = 0 ; j< 28; j++) {
                System.out.print(data[i*28 + j]);
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("Label Example");
        System.out.println(Arrays.toString(label));
        System.out.println();

    }
}
