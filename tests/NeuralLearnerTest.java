import org.json.*;
import tw.oilcast.core.*;

/** Independent deterministic sanity tests for the lightweight neural runtime. */
public final class NeuralLearnerTest {
    private static void ok(boolean condition,String message){if(!condition)throw new AssertionError(message);}
    public static void main(String[] args)throws Exception{
        double[][] inputs=new double[120][ContextData.DIM];double[] labels=new double[120];
        for(int i=0;i<inputs.length;i++){
            double first=Math.sin(i*.37),second=Math.cos(i*.19);
            inputs[i][0]=first;inputs[i][1]=second;
            labels[i]=.28*first+.1*second+.06*Math.tanh(2*first*second);
        }
        NeuralLearner.Network model=NeuralLearner.fit(inputs,labels,.001,20261028L);
        NeuralLearner.Network identical=NeuralLearner.fit(inputs,labels,.001,20261028L);
        ok(model.json().toString().equals(identical.json().toString()),"neural fitting is reproducible");
        NeuralLearner.Network restored=NeuralLearner.Network.from(model.json());
        for(int i=0;i<inputs.length;i++){
            double prediction=model.predict(inputs[i]);
            ok(Double.isFinite(prediction),"finite prediction");
            ok(restored.predict(inputs[i])==prediction,"serialized inference matches training");
        }
        double[] high=new double[ContextData.DIM],low=new double[ContextData.DIM];high[0]=1;low[0]=-1;
        ok(model.predict(high)>model.predict(low)+.12,"neural model responds to historical signals");
        boolean rejected=false;
        try{NeuralLearner.Network.from(new JSONObject().put("architecture","mlp-26-8-4-1-tanh").put("parameters",new JSONArray("[0]")));}
        catch(IllegalArgumentException expected){rejected=true;}
        ok(rejected,"malformed NN weights are rejected");
        System.out.println("PASS: deterministic deep neural training, persistence, response and input safety");
    }
}
