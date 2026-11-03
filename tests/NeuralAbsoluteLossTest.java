import tw.oilcast.core.*;

/** Uniform weekly MAE favors the conditional median, not oversampled moves. */
public final class NeuralAbsoluteLossTest {
    public static void main(String[] args){
        int n=210;double[][] x=new double[n][ContextData.DIM];double[] y=new double[n];
        for(int i=0;i<n;i++)y[i]=i%5==0?1:0;
        NeuralLearner.Network model=NeuralLearner.fit(x,y,.01,20261103);
        double prediction=model.predict(new double[ContextData.DIM]),mae=0;
        for(double actual:y)mae+=Math.abs(prediction-actual)/n;
        if(!(Math.abs(prediction)<.08&&mae<.25))throw new AssertionError("Network optimizes oversampled mean instead of weekly MAE: "+prediction+" / "+mae);
        System.out.println("NeuralAbsoluteLossTest passed: imbalanced weekly MAE="+mae+", predicted delta="+prediction);
    }
}
