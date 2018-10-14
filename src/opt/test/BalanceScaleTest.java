package opt.test;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import util.linalg.Vector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by osama on 3/6/16.
 */
public class PokerTest {

    /* Dataset Params */
    private static final String FILENAME = "src/opt/test/balance-scale.csv";
    private static int num_examples = 625;
    private static int num_attributes = 4;


    /* Randomization */
    /**
     * Randomizing your rows
     *
     * If you enable the randomization, your rows will be shuffled.
     * The seed value can be any arbitrary value. Having a seed ensures that each run of the
     * script has the same randomized shuffle ordering as any other run.
     */
    private static final boolean shouldRandomize = true;
    private static final long SEED = 0xABCDEF;

    /* Cross Validation and Testing params */
    /**
     * Set K for number of folds.
     * Set PERCENT_TRAIN for the percentage to be used for training
     */
    private static final int K = 10;
    private static final double PERCENT_TRAIN = 0.9;


    /* Neural Network Params*/
    /**
     * Tweak the following as needed. The maxLearningRate and minLearningRate are the default
     * hardcoded values in RPROPUpdateRule.java
     *
     * The accuracy threshold to stop backprop. Set this EXTREMELY low to ensure training ends only
     * when absolutely sure network output is correct
     */
    private static double initialLearningRate = 0.1;
    private static double maxLearningRate = 50;
    private static double minLearningRate = 0.000001;
    private static double backprop_threshold = 1e-10;

    /* Backprop Params */
    /**
     * If true, shouldFindNNParams determines the best params for your neural network. Since this
     * process is lengthy, you don't want to repeat it often. Do it once, record the value, and
     * place the value for numberHiddenLayerNodes.
     */
    private static final boolean shouldFindNNParams = false;
    private static int numberHiddenLayerNodes = 100;


    /* Simulated Annealing */
    /**
     * Same as above, if you've already run it once, just record the values and store them in
     * best_temp and best_cooling. Otherwise, the temps[] and cooling[] values will be cycled
     * through until the best param configuration for simulated annealing is found.
     */
//  TODO: Set this to false if you retained the best SA params from a previous run
    private static final boolean shouldFindSAParams = false;
    // TODO: Modify these to try different possible temp and cooling params
    private static double[] temps = {1E5, 1E8, 1E10, 1E12, 1E15};
    private static double[] coolingRates = {0.9, 0.95, 0.99, 0.999};

    //  TODO: Place values here from previous run if you set shouldFindSAParams to false.
    private static double best_temp = 1E8;
    private static double best_cooling = 0.99;


    /* Genetic Algorithms */
    /**
     * Same as above, if you've already run it once, just record the values and store them in
     * populationSize, toMate, and toMutate. Otherwise, the ratios will be cycled
     * through until the best param configuration for genetic algorithms is found.
     *
     * NOTE: min(populationRations) >= max(mateRatios) + max(toMutateRatio)
     * This condition must be upheld or Exceptions will be thrown later in the script
     */
    private static final boolean shouldFindGAParams = false;
    private static double[] populationRatios = {0.10, 0.15, 0.20, 0.25};
    private static double[] mateRatios = {0.02, 0.04};
    private static double[] mutateRatios = {0.02, 0.04};

    //  TODO: Place values here from previous run if you set shouldFindGAParams to false
    private static double populationRatio = 0.20;
    private static double toMateRatio = 0.04;
    private static double toMutateRatio = 0.04;


    /* Other global vars - Don't mess with these unless you have a reason to */

    // Number of input nodes is the same as the number of attributes
    private static int inputLayer = num_attributes;
    // This is determined dynamically
    private static int outputLayer;

    // Determines how many possible classifications we can have
    private static Set<String> unique_labels;

    private static Instance[] allInstances;
    private static Map<String, double[]> bitvector_mappings;

    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static DecimalFormat df = new DecimalFormat("0.000");

    // Train and Test sets
    private static DataSet trainSet;
    private static DataSet testSet;
    // Cross validation folds
    private static List<Instance[]> folds;

    // Backprop test error as baseline for plotting
    private static double backpropError;
    /* Begin actual code */

    /**
     * Comment out anything you don't want to immediately run.
     * Advice: Write loops to do certain numbers of iterations and output results for rand opt
     * algorithms
     * @param args ignored
     */
    public static void main(String[] args) throws IOException, PythonExecutionException {
        initializeInstances();

//    Creates K-folds for cross-validation
        makeTestTrainSets();
        folds = kfolds(trainSet);

        int trainIterations = 100;

//    Determine best NN params, which happen to just be the number of hidden layer nodes.
        if (shouldFindNNParams) {
            int[] hiddenLayerNodes = {5, 10, 20, 50, 100, 200};
//            trainIterations = 100;
            determineNNParams(trainIterations, hiddenLayerNodes);
        }


        /* Backprop */
        trainIterations = 100;
//    Now determine the NN performance. Results are simply printed out.
        runBackprop(trainIterations);


        /* RHC */
        // Simply run RHC as it doesn't have params
        runRHC(10000, 100, false);


//        /* SA */
//        if (shouldFindSAParams) {
//            trainIterations = 100;
//            determineSAParams(trainIterations);
//        }
        // Run actual SA with best params here
        runSA(10000, 100, best_temp, best_cooling, false);
//
//        /* GA */
//        if (shouldFindGAParams) {
//            trainIterations = 20;
//            determineGAParams(trainIterations);
//
//        }
        // Run actual GA with best params here
        runGA(10000, 100, false);
    }


    /**
     * Uses k-folds cross validation to determine the best number of hidden layers to configure a
     * nueral network among the list provided
     * @param trainIterations the number of iterations to run
     * @param hiddenLayerNodes an int[] of hidden layer nodes to try
     */
    private static void determineNNParams(int trainIterations, int[] hiddenLayerNodes) {

        System.out.println("===========Cross Validation for NN Params=========");
        double[] errors = new double[hiddenLayerNodes.length];
        for (int m = 0; m < hiddenLayerNodes.length; m++) {

            int numHiddenNodes = hiddenLayerNodes[m];

            BackPropagationNetwork[] nets = new BackPropagationNetwork[K];
            double[] validationErrors = new double[K];

//      Create and train networks
            for (int i = 0; i < nets.length; i++) {

                Instance[] validation = getValidationFold(folds, i);
                Instance[] trainFolds = getTrainFolds(folds, i);
                DataSet trnfoldsSet = new DataSet(trainFolds);

//        Creates a network with specified number of hidden layer nodes
                nets[i] = factory.createClassificationNetwork(
                        new int[]{inputLayer, numHiddenNodes, outputLayer});

                BackPropagationNetwork backpropNet = nets[i];

                ConvergenceTrainer trainer = new ConvergenceTrainer(
                        new BatchBackPropagationTrainer(trnfoldsSet, backpropNet, new SumOfSquaresError(),
                                new RPROPUpdateRule(initialLearningRate, maxLearningRate, minLearningRate)),
                        backprop_threshold, trainIterations);

                trainer.train();

                validationErrors[i] = evaluateNetwork(backpropNet, validation);
            }

//      Find the average error for this network configuration
            double error = 0;
            for (double valErr : validationErrors) {
                error += valErr;
            }
            error /= validationErrors.length;
            errors[m] = error;

            System.out.printf("Nodes: %d\tError: %s%%%n", numHiddenNodes, df.format(errors[m]));
        }

//    Find the index with the min avg validation error
        int best_index = 0;
        double minError = Double.MAX_VALUE;
        for (int j = 0; j < errors.length; j++) {
            if (errors[j] < minError) {
                minError = errors[j];
                best_index = j;
            }
        }
        int bestNumNodes = hiddenLayerNodes[best_index];

        System.out.printf("%nBest Num Hidden Nodes: %d\tError: %s%%%n", bestNumNodes, df.format
                (minError));

        numberHiddenLayerNodes = bestNumNodes;
    }


    /**
     * This method will run Backpropagation using each
     * combination of (K-1) folds for training, and the Kth fold for validation. Once the model
     * with the lowest validation set error is found, that is used as the "best" model and the
     * training and test errors on that model are recorded.
     * @param trainIterations the number of epochs/iterations
     */
    private static void runBackprop(int trainIterations) {

        System.out.println("===========Backpropagation=========");
        double starttime = System.nanoTime();
        double endtime;

        BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
                new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

        ConvergenceTrainer trainer = new ConvergenceTrainer(
                new BatchBackPropagationTrainer(trainSet, backpropNet, new SumOfSquaresError(),
                        new RPROPUpdateRule(initialLearningRate, maxLearningRate, minLearningRate)),
                backprop_threshold, trainIterations);

        trainer.train();

        double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
        double testError = evaluateNetwork(backpropNet, testSet.getInstances());

        System.out.printf("Train Error: %s%% %n", df.format(trainError));
        System.out.printf("Test Error: %s%% %n", df.format(testError));

        endtime = System.nanoTime();
        double time_elapsed = endtime - starttime;

//    Convert nanoseconds to seconds
        time_elapsed /= Math.pow(10,9);
        System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));

        // set the global variable
        backpropError = testError;

    }


    /**
     * Determines optimal weights for configured neural network using Randomized Hill Climbing with
     * Random Restarts and evaluates a neural networks performance on train and test sets with
     * those weights
     * @param trainIterations the number of iterations
     */
    private static void runRHC(int trainIterations, int step, boolean plotCurve)
            throws IOException, PythonExecutionException {

        System.out.println("===========Randomized Hill Climbing=========");

        double starttime = System.nanoTime();
        double endtime;

        BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
                new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet,
                backpropNet, measure);

        OptimizationAlgorithm oa = new RandomizedHillClimbing(nnop);

        if (plotCurve) {
            // get train/test errors to plot the learning curve,
            plotLearningCurve("Randomized Hill Climbing", trainIterations, step,
                    getErrors(oa, backpropNet, trainIterations, step));
        }
        else {
            train(oa, backpropNet, trainIterations);

            // show the training time
            endtime = System.nanoTime();
            double time_elapsed = endtime - starttime;

            // Convert nanoseconds to seconds
            time_elapsed /= Math.pow(10,9);
            System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));

            double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
            double testError = evaluateNetwork(backpropNet, testSet.getInstances());
            // display train/test error
            System.out.printf("Train Error: %s%% %n", df.format(trainError));
            System.out.printf("Test Error: %s%% %n", df.format(testError));
        }
    }


    public static void determineSAParams(int trainIterations) {

        System.out.println("===========Determining Simulated Annealing Params=========");

        double[][] errors = new double[temps.length][coolingRates.length];

        for (int x = 0; x < temps.length; x++) {

            double temp = temps[x];

            for (int y = 0; y < coolingRates.length; y++) {

                double cooling = coolingRates[y];

                BackPropagationNetwork[] nets = new BackPropagationNetwork[K];
                NeuralNetworkOptimizationProblem[] nnops = new NeuralNetworkOptimizationProblem[K];
                OptimizationAlgorithm[] oas = new OptimizationAlgorithm[K];
                double[] validationErrors = new double[K];

                for (int i = 0; i < nets.length; i++) {

                    Instance[] validation = getValidationFold(folds, i);
                    Instance[] trainFolds = getTrainFolds(folds, i);
                    DataSet trnfoldsSet = new DataSet(trainFolds);

//        Creates a network with specified number of hidden layer nodes
                    nets[i] = factory.createClassificationNetwork(
                            new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

                    BackPropagationNetwork backpropNet = nets[i];

                    nnops[i] = new NeuralNetworkOptimizationProblem(trnfoldsSet, backpropNet, measure);
                    oas[i] = new SimulatedAnnealing(temp, cooling, nnops[i]);

                    train(oas[i], nets[i], trainIterations);

                    validationErrors[i] = evaluateNetwork(backpropNet, validation);
                }

//      Find the average error for this network configuration
                double error = 0;
                for (double valErr : validationErrors) {
                    error += valErr;
                }
                error /= validationErrors.length;
                errors[x][y] = error;

            }
        }

        int best_temp_index = 0;
        int best_cool_index = 0;
        double minErr = Double.MAX_VALUE;

        for (int x = 0; x < temps.length; x++) {

            for (int y = 0; y < coolingRates.length; y++) {

                if (minErr > errors[x][y]) {
                    best_temp_index = x;
                    best_cool_index = y;
                    minErr = errors[x][y];
                }
            }
        }

        double bestCooling = coolingRates[best_cool_index];
        double bestTemp = temps[best_temp_index];
        System.out.printf("Best Cooling: %s%n", df.format(bestCooling));
        System.out.printf("Best Temp: %s%n", df.format(bestTemp));

    }


    private static void runSA(int trainIterations, int step, double temp, double cooling, boolean plotCurve)
            throws IOException, PythonExecutionException {

        System.out.println("===========Simulated Annealing=========");

        double starttime = System.nanoTime();
        double endtime;

        BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
                new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet,
                backpropNet, measure);

        OptimizationAlgorithm oa = new SimulatedAnnealing(temp, cooling, nnop);

        if (plotCurve) {
            // get train/test errors to plot the learning curve,
            plotLearningCurve("Simulated Annealing", trainIterations, step,
                    getErrors(oa, backpropNet, trainIterations, step));
        }
        else {
            train(oa, backpropNet, trainIterations);

            // show the training time
            endtime = System.nanoTime();
            double time_elapsed = endtime - starttime;

            // Convert nanoseconds to seconds
            time_elapsed /= Math.pow(10,9);
            System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));

            double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
            double testError = evaluateNetwork(backpropNet, testSet.getInstances());
            // display train/test error
            System.out.printf("Train Error: %s%% %n", df.format(trainError));
            System.out.printf("Test Error: %s%% %n", df.format(testError));
        }

    }


    public static void determineGAParams(int trainIterations) {

        System.out.println("===========Determining Genetic Algorithms Params=========");

        double[][][] errors =
                new double[populationRatios.length][mateRatios.length][mutateRatios.length];

//    Training population size is always 9/10 of the total training set, or equivalently
//    9 times the validation set
        int trainingPopulation = getValidationFold(folds, 0).length * 9;

        for (int x = 0; x < populationRatios.length; x++) {

            int population = (int) (Math.max(populationRatios[x] * trainingPopulation, 10));

            for (int y = 0; y < mateRatios.length; y++) {

                int mate = (int) (Math.max(mateRatios[y] * trainingPopulation, 10));

                for (int z = 0; z < mutateRatios.length; z++) {

                    int mutate = (int) (Math.max(mutateRatios[z] * trainingPopulation, 10));

                    BackPropagationNetwork[] nets = new BackPropagationNetwork[K];
                    NeuralNetworkOptimizationProblem[] nnops = new NeuralNetworkOptimizationProblem[K];
                    OptimizationAlgorithm[] oas = new OptimizationAlgorithm[K];
                    double[] validationErrors = new double[K];

                    for (int i = 0; i < nets.length; i++) {

                        Instance[] validation = getValidationFold(folds, i);
                        Instance[] trainFolds = getTrainFolds(folds, i);
                        DataSet trnfoldsSet = new DataSet(trainFolds);

                        nets[i] = factory.createClassificationNetwork(
                                new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

                        BackPropagationNetwork backpropNet = nets[i];

                        nnops[i] = new NeuralNetworkOptimizationProblem(trnfoldsSet, backpropNet, measure);
                        oas[i] = new StandardGeneticAlgorithm(population, mate, mutate, nnops[i]);

                        train(oas[i], backpropNet, trainIterations);

                        validationErrors[i] = evaluateNetwork(backpropNet, validation);
                    }

//      Find the average error for this configuration
                    double error = 0;
                    for (double valError : validationErrors) {
                        error += valError;
                    }
                    error /= validationErrors.length;
                    errors[x][y][z] = error;

                }

            }

        }

        int best_pop = 0;
        int best_mate = 0;
        int best_mutate = 0;
        double minErr = Double.MAX_VALUE;

        for (int x = 0; x < populationRatios.length; x++) {

            for (int y = 0; y < mateRatios.length; y++) {

                for (int z = 0; z < mutateRatios.length; z++) {

                    if (errors[x][y][z] < minErr) {
                        best_pop = x;
                        best_mate = y;
                        best_mutate = z;
                        minErr = errors[x][y][z];
                    }
                }
            }
        }

        populationRatio = populationRatios[best_pop];
        toMateRatio = mateRatios[best_mate];
        toMutateRatio = mutateRatios[best_mutate];

        System.out.printf("Population Ratio: %s%n", df.format(populationRatio));
        System.out.printf("Mate Ratio: %s%n", df.format(toMateRatio));
        System.out.printf("Mutate Ratio: %s%n", df.format(toMutateRatio));
        System.out.printf("Error: %s%% %n", df.format(minErr));

    }

    /**
     * Run genetic algorithms.
     * @param trainIterations the iterations to run
     */
    private static void runGA(int trainIterations, int step, boolean plotCurve)
            throws IOException, PythonExecutionException {

        System.out.println("===========Genetic Algorithms=========");

        double starttime = System.nanoTime();
        double endtime;

        int trainSetSize = trainSet.size();
        int populationSize = (int) (trainSetSize * populationRatio);
        int toMate = (int) (trainSetSize * toMateRatio);
        int toMutate = (int) (trainSetSize * toMutateRatio);

        BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
                new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet,
                backpropNet, measure);

        OptimizationAlgorithm oa = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, nnop);

        if (plotCurve) {
            // get train/test errors to plot the learning curve,
            plotLearningCurve("Genetic Algorithms", trainIterations, step,
                    getErrors(oa, backpropNet, trainIterations, step));
        }
        else {
            train(oa, backpropNet, trainIterations);

            // show the training time
            endtime = System.nanoTime();
            double time_elapsed = endtime - starttime;

            // Convert nanoseconds to seconds
            time_elapsed /= Math.pow(10,9);
            System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));

            double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
            double testError = evaluateNetwork(backpropNet, testSet.getInstances());
            // display train/test error
            System.out.printf("Train Error: %s%% %n", df.format(trainError));
            System.out.printf("Test Error: %s%% %n", df.format(testError));
        }
    }


    /**
     * Train a given optimization problem for a given number of iterations. Called by RHC, SA, and
     * GA algorithms
     * @param oa the optimization algorithm
     * @param network the network that corresponds to the randomized optimization problem. The
     *                optimization algorithm will determine the best weights to try using with this
     *                network and assign those weights
     * @param iterations the number of training iterations
     */
    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, int
            iterations) {

        for(int i = 0; i < iterations; i++) {
            oa.train();
        }
        Instance optimalWeights = oa.getOptimal();
        network.setWeights(optimalWeights.getData());
    }


    /**
     * Train a given optimization problem for a given number of iterations. Called by RHC, SA, and
     * GA algorithms
     * @param oa the optimization algorithm
     * @param network the network that corresponds to the randomized optimization problem. The
     *                optimization algorithm will determine the best weights to try using with this
     *                network and assign those weights
     * @param iterations the number of training iterations
     */
    private static Map<String, List<Double>> getErrors(OptimizationAlgorithm oa,
                                                       BackPropagationNetwork network,
                                                       int iterations,
                                                       int step) {
        Map<String, List<Double>> errors = new HashMap<>();
        List<Double> trainErrors = new ArrayList<>();
        List<Double> testErrors = new ArrayList<>();
        for(int i = 0; i < iterations; i++) {
            oa.train();
            if ((i % step) == 0) {
                Instance optimalWeights = oa.getOptimal();
                network.setWeights(optimalWeights.getData());
                trainErrors.add(evaluateNetwork(network, trainSet.getInstances()));
                testErrors.add(evaluateNetwork(network, testSet.getInstances()));
            }
        }
        errors.put("train", trainErrors);
        errors.put("test", testErrors);

        return errors;
    }


    /**
     * Given a network and instances, the output of the network is evaluated and a decimal value
     * for error is given
     * @param network the BackPropagationNetwork with weights already initialized
     * @param data the instances to be evaluated against
     * @return the error rate
     */
    private static double evaluateNetwork(BackPropagationNetwork network, Instance[] data) {

        double num_incorrect = 0;

        for (Instance d : data) {
            network.setInputValues(d.getData());
            network.run();

            Vector actual = d.getLabel().getData();
            Vector predicted = network.getOutputValues();


            boolean mismatch = ! isEqualOutputs(actual, predicted);

            if (mismatch) {
                num_incorrect += 1;
            }
        }

        return num_incorrect / data.length * 100;

    }


    /**
     * Compares two bit vectors to see if expected bit vector is most likely to be the same
     * class as the actual bit vector
     * @param actual the actual label
     * @param predicted the predicted label
     * @return is the prediction correct?
     */
    private static boolean isEqualOutputs(Vector actual, Vector predicted) {

        int max_at = 0;
        double max = 0;

//    Where the actual max should be
        int actual_index = 0;

        for (int i = 0; i < actual.size(); i++) {
            double aVal = actual.get(i);

            if (aVal == 1.0) {
                actual_index = i;
            }

            double bVal = predicted.get(i);

            if (bVal > max) {
                max = bVal;
                max_at = i;
            }
        }

        return actual_index == max_at;

    }


    /**
     * Reads a file formatted as CSV. Takes the labels and adds them to the set of labels (which
     * later helps determine the length of bit vectors). Records real-valued attributes. Turns the
     * attributes and labels into bit-vectors. Initializes a DataSet object with these instances.
     */
    private static void initializeInstances() {

        double[][] attributes = new double[num_examples][];

        String[] labels = new String[num_examples];
        unique_labels = new HashSet<>();


//    Reading dataset
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(FILENAME)));

//      You don't need these headers, they're just the column labels

//            String useless_headers = br.readLine();

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[num_attributes];

                for(int j = 0; j < num_attributes; j++) {
                    attributes[i][j] = Double.parseDouble(scan.next());
                }

//        This last element is actually your classification, which is assumed to be a string
                labels[i] = scan.next();
                unique_labels.add(labels[i]);
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }


//    Creating a mapping of bitvectors. So "some classification" => [0, 1, 0, 0]
        int distinct_labels = unique_labels.size();
        outputLayer = distinct_labels;

        bitvector_mappings = new HashMap<>();

        int index = 0;
        for (String label : unique_labels) {
            double[] bitvect = new double[distinct_labels];

//      At index, set to 1 for a given string
            bitvect[index] = 1.0;
//      Increment which index will have a bit flipped in next classification
            index++;

            bitvector_mappings.put(label, bitvect);
        }

//    Replaces the label for each instance with the corresponding bit vector for that label
//    This works even for binary classification
        allInstances = new Instance[num_examples];
        for (int i = 0; i < attributes.length; i++) {
            double[] X = attributes[i];

            String label = labels[i];
            double[] bitvect = bitvector_mappings.get(label);

            Instance instance = new Instance(X);
            instance.setLabel(new Instance(bitvect));

            allInstances[i] = instance;
        }
    }


    /**
     * Print out the actual vs expected bit-vector. Used for debugging purposes only
     * @param actual what the example's actual bit vector looks like
     * @param expected what a network output as a bit vector
     */
    public static void printVectors(Vector actual, Vector expected) {
        System.out.print("Actual: [");
        for (int i = 0; i < actual.size(); i++) {
            System.out.printf(" %f", actual.get(i));
        }
        System.out.print(" ] \t Expected: [");

        for (int i = 0; i < expected.size(); i++) {
            System.out.printf(" %f", expected.get(i));
        }
        System.out.println(" ]");
    }


    /**
     * Takes all instances, and randomly orders them. Then, the first PERCENT_TRAIN percentage of
     * instances form the trainSet DataSet, and the remaining (1 - PERCENT_TRAIN) percentage of
     * instances form the testSet DataSet.
     */
    private static void makeTestTrainSets() {

        List<Instance> instances = new ArrayList<>();

        for (Instance instance: allInstances) {
            instances.add(instance);
        }

        Random rand = new Random(SEED);
        if (shouldRandomize) {
            Collections.shuffle(instances, rand);
        }

        int cutoff = (int) (instances.size() * PERCENT_TRAIN);

        List<Instance> trainInstances = instances.subList(0, cutoff);
        List<Instance> testInstances = instances.subList(cutoff, instances.size());

        Instance[] arr_trn = new Instance[trainInstances.size()];
        trainSet = new DataSet(trainInstances.toArray(arr_trn));

//    System.out.println("Train Set ")

        Instance[] arr_tst = new Instance[testInstances.size()];
        testSet = new DataSet(testInstances.toArray(arr_tst));

    }


    /**
     * Given a DataSet of training data, separate the instances into K nearly-equal-sized
     * partitions called folds for K-folds cross validation
     * @param training, the training DataSet
     * @return a list of folds, where each fold is an Instance[]
     */
    private static List<Instance[]> kfolds(DataSet training) {

        Instance[] trainInstances = training.getInstances();

        List<Instance> instances = new ArrayList<>();
        for (Instance instance: trainInstances) {
            instances.add(instance);
        }

        List<Instance[]> folds = new ArrayList<>();

//    Number of values per fold
        int per_fold = (int) Math.floor((double)(instances.size()) / K);

        int start = 0;
        int end = per_fold;

        while (start < instances.size()) {


            List<Instance> foldList = null;

            if (end > instances.size()) {
                end = instances.size();
            }
            foldList = instances.subList(start, end);

            Instance[] fold = new Instance[foldList.size()];
            fold = foldList.toArray(fold);

            folds.add(fold);

            start = end + 1;
            end = start + per_fold;

        }
        return folds;
    }


    /**
     * Given a list of Instance[], this helper combines each arrays contents into one, single
     * output array
     * @param instanceList the list of Instance[]
     * @return the combined array consisting of the contents of each Instance[] in instanceList
     */
    private static Instance[] combineInstances(List<Instance[]> instanceList) {
        List<Instance> combined = new ArrayList<>();

        for (Instance[] fold: instanceList) {

            for (Instance instance : fold) {
                combined.add(instance);
            }
        }

        Instance[] combinedArr = new Instance[combined.size()];
        combinedArr = combined.toArray(combinedArr);
        return combinedArr;
    }


    /**
     * Given a list of folds and an index, it will provide an Instance[] with the combined
     * instances from every fold except for the fold at the given index
     * @param folds the K-folds, a list of Instance[] used as folds for cross-validation
     * @param foldIndex the index of the fold to exclude. That fold is used as the validation set
     * @return the training folds combined into once Instance[]
     */
    private static Instance[] getTrainFolds(List<Instance[]> folds, int foldIndex) {
        List<Instance[]> trainFolds = new ArrayList<>(folds);
        trainFolds.remove(foldIndex);

        Instance[] trnfolds = combineInstances(trainFolds);
        return trnfolds;
    }


    /**
     * Given a list of folds and an index, it will provide an Instance[] to serve as a validation
     * set.
     * @param folds the K-folds, a list of Instance[] used as folds for cross-validation
     * @param foldIndex the index of the fold to use as the validation set
     * @return the validation set
     */
    private static Instance[] getValidationFold(List<Instance[]> folds, int foldIndex) {
        return folds.get(foldIndex);
    }

    /**
     * This method runs the optimization algorithm given a series of iterations
     * and stores both train and test set error for generating learning curve
     * against iterations.
     * The matplotlib4j library is used to plot learning curves. It is similar to Python's
     * matplotlib with a similar API.
     * @param algName the name of the optimization algorithm used
     * @param errors the number of iterations to run against
     */
    private static void plotLearningCurve(String algName, int trainIterations, int step,
                                         Map<String, List<Double>> errors)
            throws IOException, PythonExecutionException {
        System.out.println("===========" + algName + "=========");

        List<Double> trainErrors = errors.get("train");
        List<Double> testErrors = errors.get("test");

        System.out.printf("Train Error: %s%% %n",
                df.format(trainErrors.get(trainErrors.size() - 1)));
        System.out.printf("Test Error: %s%% %n",
                df.format(testErrors.get(testErrors.size() - 1)));

        List<Integer> xlabels = IntStream.range(1, trainIterations / step + 1).map(x -> x * step)
                .boxed().collect(Collectors.toList());
        List<Double> baseline = Collections.nCopies(xlabels.size(), backpropError);  // baseline

        Plot plt = Plot.create();
        // add training curve
        plt.plot()
                .add(xlabels, trainErrors)
                .label("train")
                .linestyle("-");
        // add test curve
        plt.plot()
                .add(xlabels, testErrors)
                .label("test")
                .linestyle("-");
        // add baseline
        plt.plot()
                .add(xlabels, baseline)
                .label("baseline: Back-propagation")
                .linestyle("--");
        plt.xlabel("Iterations");
        plt.ylabel("Error (%)");
//        plt.text(0.5, 0.2, "text");
        plt.title("Learning Curve - " + algName);
        plt.legend();
//        plt.show();
        plt.savefig("src/opt/test/" + algName + ".png").dpi(200);
        plt.executeSilently();
    }

}