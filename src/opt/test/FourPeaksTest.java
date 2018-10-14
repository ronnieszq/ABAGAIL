package opt.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
import opt.ga.SingleCrossOver;
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
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    private static List<Integer> itersRHC = new ArrayList<>();
    private static List<Integer> itersSA = new ArrayList<>();
    private static List<Integer> itersGA = new ArrayList<>();
    private static List<Integer> itersMIMIC = new ArrayList<>();
    
    private static void runFourPeaks(int n) {
        int t = n / 10;
        int[] ranges = new int[n];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(t);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        // RHC
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
//        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        ConvergenceTrainer fit = new ConvergenceTrainer(rhc);
        fit.train();
        System.out.println("RHC: " + ef.value(rhc.getOptimal()));
        itersRHC.add(fit.getIterations());
        System.out.println("RHC Iterations: " + fit.getIterations());
        System.out.println("============================");

        // SA
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
//        fit = new FixedIterationTrainer(sa, 200000);
        fit = new ConvergenceTrainer(sa);
        fit.train();
        System.out.println("SA: " + ef.value(sa.getOptimal()));
        itersSA.add(fit.getIterations());
        System.out.println("SA Iterations: " + fit.getIterations());
        System.out.println("============================");

        // GA
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
//        fit = new FixedIterationTrainer(ga, 1000);
        fit = new ConvergenceTrainer(ga);
        fit.train();
        System.out.println("GA: " + ef.value(ga.getOptimal()));
        itersGA.add(fit.getIterations());
        System.out.println("GA Iterations: " + fit.getIterations());
        System.out.println("============================");

        // MIMIC
        MIMIC mimic = new MIMIC(200, 20, pop);
//        fit = new FixedIterationTrainer(mimic, 1000);
        fit = new ConvergenceTrainer(mimic);
        fit.train();
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
        itersMIMIC.add(fit.getIterations());
        System.out.println("MIMIC Iterations: " + fit.getIterations());

    }

    public static void main(String[] args) throws IOException, PythonExecutionException {
        List<Integer> nCities = IntStream.range(3, 10).map(x -> x * 10)
                .boxed().collect(Collectors.toList());
        for (int i : nCities) { runFourPeaks(i); }

        // fitness curve
        Plot plt = Plot.create();
        // add RHC curve
        plt.plot()
                .add(nCities, itersRHC)
                .label("RHC")
                .linestyle("-");
        // add SA curve
        plt.plot()
                .add(nCities, itersSA)
                .label("SA")
                .linestyle("-");
        // add GA curve
        plt.plot()
                .add(nCities, itersGA)
                .label("GA")
                .linestyle("-");
        // add MIMIC curve
        plt.plot()
                .add(nCities, itersMIMIC)
                .label("MIMIC")
                .linestyle("-");
        plt.xlabel("Number of Cities");
        plt.ylabel("Travel Distance");
        plt.title("Shortest Distance");
        plt.legend();
        plt.show();
//        plt.savefig("src/opt/test/tsp-fitness.png").dpi(200);
//        plt.executeSilently();
    }
}
