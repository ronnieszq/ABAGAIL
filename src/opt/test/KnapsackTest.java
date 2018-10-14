package opt.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.ConvergenceTrainer;
import shared.FixedIterationTrainer;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;

    private static List<Integer> itersRHC = new ArrayList<>();
    private static List<Double> timesRHC = new ArrayList<>();
    private static List<Double> valuesRHC = new ArrayList<>();
    private static List<Integer> itersSA = new ArrayList<>();
    private static List<Double> timesSA = new ArrayList<>();
    private static List<Double> valuesSA = new ArrayList<>();
    private static List<Integer> itersGA = new ArrayList<>();
    private static List<Double> timesGA = new ArrayList<>();
    private static List<Double> valuesGA = new ArrayList<>();
    private static List<Integer> itersMIMIC = new ArrayList<>();
    private static List<Double> timesMIMIC = new ArrayList<>();
    private static List<Double> valuesMIMIC = new ArrayList<>();

    private static void runKnapsack(int n) {
        double MAX_KNAPSACK_WEIGHT = MAX_WEIGHT * n * COPIES_EACH * .4;
        int[] copies = new int[n];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[n];
        double[] weights = new double[n];
        for (int i = 0; i < n; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[n];
        Arrays.fill(ranges, COPIES_EACH + 1);

        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        // RHC
        double starttime = System.currentTimeMillis();
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
//        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        ConvergenceTrainer fit = new ConvergenceTrainer(rhc);
        fit.train();
        double timeElapse = System.currentTimeMillis() - starttime;
        timesRHC.add(timeElapse);
        System.out.println("RHC Time : "+ timeElapse);
        valuesRHC.add(ef.value(rhc.getOptimal()));
        System.out.println("RHC Fitness: " + ef.value(rhc.getOptimal()));
        itersRHC.add(fit.getIterations());
        System.out.println("RHC Iterations: " + fit.getIterations());
        System.out.println("============================");

        // SA
        starttime = System.currentTimeMillis();
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
//        fit = new FixedIterationTrainer(sa, 200000);
        fit = new ConvergenceTrainer(sa);
        fit.train();
        timeElapse = System.currentTimeMillis() - starttime;
        timesSA.add(timeElapse);
        System.out.println("SA Time : "+ timeElapse);
        itersSA.add(fit.getIterations());
        System.out.println("SA Iterations: " + fit.getIterations());
        valuesSA.add(ef.value(sa.getOptimal()));
        System.out.println("SA Fitness: " + ef.value(sa.getOptimal()));
//        System.out.println("SA: Board Position: ");
//        System.out.println(ef.boardPositions());
        System.out.println("============================");

        // GA
        starttime = System.currentTimeMillis();
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
//        fit = new FixedIterationTrainer(ga, 1000);
        fit = new ConvergenceTrainer(ga);
        fit.train();
        timeElapse = System.currentTimeMillis() - starttime;
        timesGA.add(timeElapse);
        System.out.println("GA Time : "+ timeElapse);
        itersGA.add(fit.getIterations());
        System.out.println("GA Iterations: " + fit.getIterations());
        valuesGA.add(ef.value(ga.getOptimal()));
        System.out.println("GA Fitness: " + ef.value(ga.getOptimal()));
//        System.out.println("GA: Board Position: ");
//        System.out.println(ef.boardPositions());
        System.out.println("============================");

        // MIMIC
        starttime = System.currentTimeMillis();
        MIMIC mimic = new MIMIC(200, 100, pop);
//        fit = new FixedIterationTrainer(mimic, 1000);
        fit = new ConvergenceTrainer(mimic);
        fit.train();
        timeElapse = System.currentTimeMillis() - starttime;
        timesMIMIC.add(timeElapse);
        System.out.println("MIMIC Time : "+ timeElapse);
        itersMIMIC.add(fit.getIterations());
        System.out.println("MIMIC Iterations: " + fit.getIterations());
        valuesMIMIC.add(ef.value(mimic.getOptimal()));
        System.out.println("MIMIC Fitness: " + ef.value(mimic.getOptimal()));
    }

    public static void main(String[] args) throws IOException, PythonExecutionException {
        List<Integer> nItems = IntStream.range(4, 21).map(x -> x * 10)
                .boxed().collect(Collectors.toList());
        for (int i : nItems) { runKnapsack(i); }

        // values curve
        Plot plt = Plot.create();
        // add RHC curve
        plt.plot()
                .add(nItems, valuesRHC)
                .label("RHC")
                .linestyle("-");
        // add SA curve
        plt.plot()
                .add(nItems, valuesSA)
                .label("SA")
                .linestyle("-");
        // add GA curve
        plt.plot()
                .add(nItems, valuesGA)
                .label("GA")
                .linestyle("-");
        // add MIMIC curve
        plt.plot()
                .add(nItems, valuesMIMIC)
                .label("MIMIC")
                .linestyle("-");
        plt.xlabel("Number of Items");
        plt.ylabel("Knapsack Weight");
        plt.title("Largest Value of the Knapsack");
        plt.legend();
//        plt.show();
        plt.savefig("src/opt/test/knapsack-value.png").dpi(200);
        plt.executeSilently();

        // iterations curve
        plt = Plot.create();
        // add RHC curve
        plt.plot()
                .add(nItems, itersRHC)
                .label("RHC")
                .linestyle("-");
        // add SA curve
        plt.plot()
                .add(nItems, itersSA)
                .label("SA")
                .linestyle("-");
        // add GA curve
        plt.plot()
                .add(nItems, itersGA)
                .label("GA")
                .linestyle("-");
        // add MIMIC curve
        plt.plot()
                .add(nItems, itersMIMIC)
                .label("MIMIC")
                .linestyle("-");
        plt.xlabel("Number of Items");
        plt.ylabel("Number of Evaluations");
        plt.title("Function Evaluations Required to Converge");
        plt.legend();
//        plt.show();
        plt.savefig("src/opt/test/knapsack-evaluation.png").dpi(200);
        plt.executeSilently();
    }

}
