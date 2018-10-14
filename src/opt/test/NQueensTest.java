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
import opt.ga.NQueensFitnessFunction;
import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import shared.ConvergenceTrainer;

/**
 * @author kmanda1
 * @version 1.0
 */
public class NQueensTest {

    private static List<Integer> itersRHC = new ArrayList<>();
    private static List<Double> timesRHC = new ArrayList<>();
    private static List<Integer> itersSA = new ArrayList<>();
    private static List<Double> timesSA = new ArrayList<>();
    private static List<Integer> itersGA = new ArrayList<>();
    private static List<Double> timesGA = new ArrayList<>();
    private static List<Integer> itersMIMIC = new ArrayList<>();
    private static List<Double> timesMIMIC = new ArrayList<>();
    
    private static void runNQueens(int n) {
        int[] ranges = new int[n];
        Random random = new Random(n);
        for (int i = 0; i < n; i++) {
        	ranges[i] = random.nextInt();
        }
        NQueensFitnessFunction ef = new NQueensFitnessFunction();
        Distribution odd = new DiscretePermutationDistribution(n);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        double starttime = System.currentTimeMillis();
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
//        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 100);
        ConvergenceTrainer fit = new ConvergenceTrainer(rhc);
        fit.train();
        double timeElapse = System.currentTimeMillis() - starttime;
        timesRHC.add(timeElapse);
        System.out.println("RHC Time : "+ timeElapse);
        itersRHC.add(fit.getIterations());
        System.out.println("RHC Iterations: " + fit.getIterations());
        System.out.println("RHC Fitness: " + ef.value(rhc.getOptimal()));
//        System.out.println("RHC: Board Position: ");
//        System.out.println(ef.boardPositions());
        System.out.println("============================");

        starttime = System.currentTimeMillis();
        SimulatedAnnealing sa = new SimulatedAnnealing(1E1, .1, hcp);
//        fit = new FixedIterationTrainer(sa, 100);
        fit = new ConvergenceTrainer(sa);
        fit.train();
        timeElapse = System.currentTimeMillis() - starttime;
        timesSA.add(timeElapse);
        System.out.println("SA Time : "+ timeElapse);
        itersSA.add(fit.getIterations());
        System.out.println("SA Iterations: " + fit.getIterations());
        System.out.println("SA Fitness: " + ef.value(sa.getOptimal()));
//        System.out.println("SA: Board Position: ");
//        System.out.println(ef.boardPositions());
        System.out.println("============================");

        starttime = System.currentTimeMillis();
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 0, 10, gap);
//        fit = new FixedIterationTrainer(ga, 100);
        fit = new ConvergenceTrainer(ga);
        fit.train();
        timeElapse = System.currentTimeMillis() - starttime;
        timesGA.add(timeElapse);
        System.out.println("GA Time : "+ timeElapse);
        itersGA.add(fit.getIterations());
        System.out.println("GA Iterations: " + fit.getIterations());
        System.out.println("GA Fitness: " + ef.value(ga.getOptimal()));
//        System.out.println("GA: Board Position: ");
//        System.out.println(ef.boardPositions());
        System.out.println("============================");

        starttime = System.currentTimeMillis();
        MIMIC mimic = new MIMIC(200, 10, pop);
//        fit = new FixedIterationTrainer(mimic, 5);
        fit = new ConvergenceTrainer(mimic);
        fit.train();
        timeElapse = System.currentTimeMillis() - starttime;
        timesMIMIC.add(timeElapse);
        System.out.println("MIMIC Time : "+ timeElapse);
        itersMIMIC.add(fit.getIterations());
        System.out.println("MIMIC Iterations: " + fit.getIterations());
        System.out.println("MIMIC Fitness: " + ef.value(mimic.getOptimal()));
//        System.out.println("MIMIC: Board Position: ");
//        System.out.println(ef.boardPositions());
    }

    public static void main(String[] args) throws IOException, PythonExecutionException {
        List<Integer> nQueens = IntStream.range(2, 10).boxed().collect(Collectors.toList());
        for (int i : nQueens) { runNQueens((int) Math.pow(2, i)); }

        // iteration curve
        Plot plt = Plot.create();
        // add RHC curve
        plt.plot()
                .add(nQueens, itersRHC)
                .label("RHC")
                .linestyle("-");
        // add SA curve
        plt.plot()
                .add(nQueens, itersSA)
                .label("SA")
                .linestyle("-");
        // add GA curve
        plt.plot()
                .add(nQueens, itersGA)
                .label("GA")
                .linestyle("-");
        // add MIMIC curve
        plt.plot()
                .add(nQueens, itersMIMIC)
                .label("MIMIC")
                .linestyle("-");
        plt.xlabel("Number of Queens");
        plt.ylabel("Log of Evaluations");
        plt.title("Function Evaluations Required to Converge");
        plt.legend();
//        plt.show();
        plt.savefig("src/opt/test/nqueens-evaluations.png").dpi(200);
        plt.executeSilently();

        // time curve
        plt = Plot.create();
        // add RHC curve
        plt.plot()
                .add(nQueens, timesRHC)
                .label("RHC")
                .linestyle("-");
        // add SA curve
        plt.plot()
                .add(nQueens, timesSA)
                .label("SA")
                .linestyle("-");
        // add GA curve
        plt.plot()
                .add(nQueens, timesGA)
                .label("GA")
                .linestyle("-");
        // add MIMIC curve
        plt.plot()
                .add(nQueens, timesMIMIC)
                .label("MIMIC")
                .linestyle("-");
        plt.xlabel("Number of Queens");
        plt.ylabel("Time in Millisecond");
        plt.title("Time to Converge");
        plt.legend();
//        plt.show();
        plt.savefig("src/opt/test/nqueens-runtime.png").dpi(200);
        plt.executeSilently();
    }
}
