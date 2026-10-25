package org.example.deeplearning;
import org.example.deeplearning.activation.ActivationFunction;
import org.example.deeplearning.core.NeuralNetwork;
import org.example.deeplearning.evaluation.ClassificationMetrics;
import org.example.deeplearning.initialization.WeightInitializer;
import org.example.deeplearning.loss.*;
import org.example.deeplearning.optim.RMSProp;
import org.example.deeplearning.training.Trainer;
import java.util.Random;
public final class ExpansionChecks {
    private static void close(double a, double b, double tolerance) {
        if (!Double.isFinite(a) || Math.abs(a-b)>tolerance) throw new AssertionError(a+" != "+b);
    }
    private static void checkGradient(LossFunction loss, double[] p, double[] t) {
        double[] g=loss.gradient(p,t); double e=1e-6;
        for (int i=0;i<p.length;i++) {
            double old=p[i]; p[i]=old+e; double a=loss.loss(p,t);
            p[i]=old-e; double b=loss.loss(p,t); p[i]=old;
            close(g[i],(a-b)/(2*e),1e-7);
        }
    }
    public static void main(String[] args) {
        checkGradient(new MeanAbsoluteError(),new double[]{.3,-2},new double[]{1,0});
        checkGradient(new HuberLoss(1),new double[]{.3,-2},new double[]{1,0});
        close(new MeanAbsoluteError().gradient(new double[]{1},new double[]{1})[0],0,0);
        close(new HuberLoss(1).loss(new double[]{.5,2},new double[]{0,0}),.8125,1e-12);
        RMSProp opt=new RMSProp(.1,.5,1e-8); double[] p={1};
        opt.update(p,new double[]{2}); close(p[0],1-.2/(Math.sqrt(2)+1e-8),1e-12);
        opt.update(p,new double[]{2}); close(p[0],1-.2/(Math.sqrt(2)+1e-8)-.2/(Math.sqrt(3)+1e-8),1e-12);
        close(WeightInitializer.heNormal().sample(3,4,new Random(7)),WeightInitializer.heNormal().sample(3,4,new Random(7)),0);
        int[][] m=ClassificationMetrics.confusionMatrix(new int[]{0,1,1},new int[]{0,0,1},2);
        if(m[0][0]!=1||m[1][0]!=1||m[1][1]!=1) throw new AssertionError("confusion matrix");
        close(ClassificationMetrics.accuracy(new int[]{0,1,1},new int[]{0,0,1}),2.0/3,1e-12);
        double[][] x={{-2},{-1},{0},{1},{2}}, y={{-3},{-1},{1},{3},{5}};
        NeuralNetwork n=new NeuralNetwork(new int[]{1,1},new ActivationFunction[]{ActivationFunction.linear(),ActivationFunction.linear()},new HuberLoss(1),new RMSProp(.01),42,WeightInitializer.xavierUniform());
        double before=n.calculateLoss(x,y); n.fit(new Trainer(1000,5,true,42),x,y);
        if(n.calculateLoss(x,y)>=before*.01) throw new AssertionError("regression did not converge");
        System.out.println("Expansion checks passed");
    }
}
