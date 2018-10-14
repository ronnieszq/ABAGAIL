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
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.ConvergenceTrainer;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    private static List<Integer> itersRHC = new ArrayList<>();
    private static List<Double> timesRHC = new ArrayList<>();
    private static List<Double> fitnessesRHC = new ArrayList<>();
    private static List<Integer> itersSA = new ArrayList<>();
    private static List<Double> timesSA = new ArrayList<>();
    private static List<Double> fitnessesSA = new ArrayList<>();
    private static List<Integer> itersGA = new ArrayList<>();
    private static List<Double> timesGA = new ArrayList<>();
    private static List<Double> fitnessesGA = new ArrayList<>();
    private static List<Integer> itersMIMIC = new ArrayList<>();
    private static List<Double> timesMIMIC = new ArrayList<>();
    private static List<Double> fitnessesMIMIC = new ArrayList<>();

    private static void runTSP(int n) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[n][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(n);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        // RHC
        double starttime = System.currentTimeMillis();
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
//        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        ConvergenceTrainer fit = new ConvergenceTrainer(rhc);
        fit.train();
        double timeElapse = System.currentTimeMillis() - starttime;
        timesRHC.add(timeElapse);
        System.out.println("RHC Time : "+ timeElapse);
        fitnessesRHC.add(ef.value(rhc.getOptimal()));
        System.out.println("RHC Fitness: " + ef.value(rhc.getOptimal()));
        itersRHC.add(fit.getIterations());
        System.out.println("RHC Iterations: " + fit.getIterations());
        System.out.println("============================");

        // SA
        starttime = System.currentTimeMillis();
        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
//        fit = new FixedIterationTrainer(sa, 200000);
        fit = new ConvergenceTrainer(sa);
        fit.train();
        timeElapse = System.currentTimeMillis() - starttime;
        timesSA.add(timeElapse);
        System.out.println("SA Time : "+ timeElapse);
        itersSA.add(fit.getIterations());
        System.out.println("SA Iterations: " + fit.getIterations());
        fitnessesSA.add(ef.value(sa.getOptimal()));
        System.out.println("SA Fitness: " + ef.value(sa.getOptimal()));
//        System.out.println("SA: Board Position: ");
//        System.out.println(ef.boardPositions());
        System.out.println("============================");

        // GA
        starttime = System.currentTimeMillis();
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
//        fit = new FixedIterationTrainer(ga, 1000);
        fit = new ConvergenceTrainer(ga);
        fit.train();
        timeElapse = System.currentTimeMillis() - starttime;
        timesGA.add(timeElapse);
        System.out.println("GA Time : "+ timeElapse);
        itersGA.add(fit.getIterations());
        System.out.println("GA Iterations: " + fit.getIterations());
        fitnessesGA.add(ef.value(ga.getOptimal()));
        System.out.println("GA Fitness: " + ef.value(ga.getOptimal()));
//        System.out.println("GA: Board Position: ");
//        System.out.println(ef.boardPositions());
        System.out.println("============================");

        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[n];
        Arrays.fill(ranges, n);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

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
        fitnessesMIMIC.add(ef.value(mimic.getOptimal()));
        System.out.println("MIMIC Fitness: " + ef.value(mimic.getOptimal()));
//        System.out.println("MIMIC: Board Position: ");
//        System.out.println(ef.boardPositions());
        
    }

    public static void main(String[] args) throws IOException, PythonExecutionException {
        List<Integer> nCities = IntStream.range(1, 11).map(x -> x * 10)
                .boxed().collect(Collectors.toList());
        for (int i : nCities) { runTSP(i); }

        // fitness curve
        Plot plt = Plot.create();
        // add RHC curve
        plt.plot()
                .add(nCities, fitnessesRHC)
                .label("RHC")
                .linestyle("-");
        // add SA curve
        plt.plot()
                .add(nCities, fitnessesSA)
                .label("SA")
                .linestyle("-");
        // add GA curve
        plt.plot()
                .add(nCities, fitnessesGA)
                .label("GA")
                .linestyle("-");
        // add MIMIC curve
        plt.plot()
                .add(nCities, fitnessesMIMIC)
                .label("MIMIC")
                .linestyle("-");
        plt.xlabel("Number of Cities");
        plt.ylabel("Travel Distance");
        plt.title("Shortest Distance");
        plt.legend();
//        plt.show();
        plt.savefig("src/opt/test/tsp-fitness.png").dpi(200);
        plt.executeSilently();

        // time curve
        plt = Plot.create();
        // add RHC curve
        plt.plot()
                .add(nCities, timesRHC)
                .label("RHC")
                .linestyle("-");
        // add SA curve
        plt.plot()
                .add(nCities, timesSA)
                .label("SA")
                .linestyle("-");
        // add GA curve
        plt.plot()
                .add(nCities, timesGA)
                .label("GA")
                .linestyle("-");
        // add MIMIC curve
        plt.plot()
                .add(nCities, timesMIMIC)
                .label("MIMIC")
                .linestyle("-");
        plt.xlabel("Number of Cities");
        plt.ylabel("Time in Millisecond");
        plt.title("Runtime");
        plt.legend();
//        plt.show();
        plt.savefig("src/opt/test/tsp-runtime.png").dpi(200);
        plt.executeSilently();
    }
}
