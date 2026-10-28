package tw.oilcast.core;

import org.json.*;
import java.util.*;

/** Small, deterministic two-hidden-layer neural network for weekly price changes.
 *  No external ML runtime is needed for Android inference. */
public final class NeuralLearner {
    private NeuralLearner(){}
    public static final int INPUT=ContextData.DIM, H1=8, H2=4;
    private static final int W1=0, B1=H1*INPUT, W2=B1+H1, B2=W2+H2*H1,
        W3=B2+H2, B3=W3+H2, SIZE=B3+1;
    private static final int MAX_EPOCHS=110, PATIENCE=15;

    public static final class Network {
        private final double[] theta;
        public Network(double[] values){
            if(values.length!=SIZE)throw new IllegalArgumentException("神經網路維度錯誤");
            theta=values.clone();
            for(double v:theta)if(!Double.isFinite(v)||Math.abs(v)>20)throw new IllegalArgumentException("神經網路權重異常");
        }
        public double predict(double[] x){
            if(x.length!=INPUT)throw new IllegalArgumentException("神經網路輸入維度錯誤");
            double[] h1=new double[H1],h2=new double[H2];
            for(int i=0;i<H1;i++){
                double z=theta[B1+i];for(int j=0;j<INPUT;j++)z+=theta[W1+i*INPUT+j]*x[j];h1[i]=Math.tanh(z);
            }
            for(int i=0;i<H2;i++){
                double z=theta[B2+i];for(int j=0;j<H1;j++)z+=theta[W2+i*H1+j]*h1[j];h2[i]=Math.tanh(z);
            }
            double y=theta[B3];for(int i=0;i<H2;i++)y+=theta[W3+i]*h2[i];
            if(!Double.isFinite(y))throw new IllegalArgumentException("神經網路預測異常");return y;
        }
        public JSONObject json()throws JSONException{return new JSONObject().put("architecture","mlp-26-8-4-1-tanh")
            .put("parameters",Learner.array(theta));}
        public static Network from(JSONObject j)throws JSONException{
            if(!"mlp-26-8-4-1-tanh".equals(j.getString("architecture")))throw new IllegalArgumentException("神經網路架構不符");
            JSONArray a=j.getJSONArray("parameters");if(a.length()!=SIZE)throw new IllegalArgumentException("神經網路參數數量錯誤");
            double[] p=new double[SIZE];for(int i=0;i<SIZE;i++)p[i]=a.getDouble(i);return new Network(p);
        }
    }
    private static double[] init(long seed){
        Random rng=new Random(seed);double[] p=new double[SIZE];
        for(int i=W1;i<B1;i++)p[i]=rng.nextGaussian()*.48/Math.sqrt(INPUT);
        for(int i=W2;i<B2;i++)p[i]=rng.nextGaussian()*.48/Math.sqrt(H1);
        for(int i=W3;i<B3;i++)p[i]=rng.nextGaussian()*.12/Math.sqrt(H2);
        return p;
    }
    /** Full-batch Adam with recency weights, shrinkage and chronological early stopping. */
    public static Network fit(double[][] x,double[] y,double decay,long seed){
        if(x.length!=y.length||x.length<32||decay<0||decay>.1)throw new IllegalArgumentException("神經網路訓練樣本異常");
        int hold=Math.max(12,x.length/7),train=x.length-hold;
        double[] p=init(seed),best=p.clone(),m=new double[SIZE],v=new double[SIZE];
        double bestLoss=Double.POSITIVE_INFINITY;int chosen=1,stale=0;
        for(int epoch=1;epoch<=MAX_EPOCHS;epoch++){
            step(p,m,v,x,y,train,decay,epoch);
            double loss=score(p,x,y,train,x.length);
            if(loss+1e-7<bestLoss){bestLoss=loss;best=p.clone();chosen=epoch;stale=0;}
            else if(++stale>=PATIENCE)break;
        }
        // Retrain for the selected number of epochs using *all* observations available at this origin.
        p=init(seed);m=new double[SIZE];v=new double[SIZE];
        for(int epoch=1;epoch<=chosen;epoch++)step(p,m,v,x,y,x.length,decay,epoch);
        return new Network(p);
    }
    private static double score(double[] p,double[][] x,double[] y,int start,int end){
        Network model=new Network(p);double loss=0,weight=0;
        for(int i=start;i<end;i++){
            double w=Math.pow(.5,(end-1-i)/26.0)*(Math.abs(y[i])>=.049?2.5:1);
            loss+=w*Math.abs(model.predict(x[i])-y[i]);weight+=w;
        }
        return loss/Math.max(1e-9,weight);
    }
    private static void step(double[] p,double[] m,double[] v,double[][] x,double[] y,int count,double decay,int epoch){
        double[] grad=new double[SIZE];double total=0;
        for(int row=0;row<count;row++){
            double w=Math.pow(.5,(count-1-row)/26.0)*(Math.abs(y[row])>=.049?2.5:1);total+=w;double[] z=x[row];
            double[] a=new double[H1],b=new double[H2];
            for(int i=0;i<H1;i++){double t=p[B1+i];for(int j=0;j<INPUT;j++)t+=p[W1+i*INPUT+j]*z[j];a[i]=Math.tanh(t);}
            for(int i=0;i<H2;i++){double t=p[B2+i];for(int j=0;j<H1;j++)t+=p[W2+i*H1+j]*a[j];b[i]=Math.tanh(t);}
            double prediction=p[B3];for(int i=0;i<H2;i++)prediction+=p[W3+i]*b[i];
            double d=2*w*Math.max(-2,Math.min(2,prediction-y[row]));grad[B3]+=d;
            double[] db=new double[H2],da=new double[H1];
            for(int i=0;i<H2;i++){grad[W3+i]+=d*b[i];db[i]=d*p[W3+i]*(1-b[i]*b[i]);grad[B2+i]+=db[i];}
            for(int i=0;i<H2;i++)for(int j=0;j<H1;j++){grad[W2+i*H1+j]+=db[i]*a[j];da[j]+=db[i]*p[W2+i*H1+j];}
            for(int i=0;i<H1;i++){da[i]*=1-a[i]*a[i];grad[B1+i]+=da[i];for(int j=0;j<INPUT;j++)grad[W1+i*INPUT+j]+=da[i]*z[j];}
        }
        double lr=.022;
        for(int i=0;i<SIZE;i++){
            boolean weight=i<B1||i>=W2&&i<B2||i>=W3&&i<B3;
            double g=grad[i]/total+(weight?decay*p[i]:0);
            m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g;
            p[i]-=lr*(m[i]/(1-Math.pow(.9,epoch)))/(Math.sqrt(v[i]/(1-Math.pow(.999,epoch)))+1e-8);
        }
    }
}
