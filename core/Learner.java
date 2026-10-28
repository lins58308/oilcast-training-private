package tw.oilcast.core;

import java.util.*;
import org.json.*;

/** Small on-device ridge autoregression. All walk-forward fits see past observations only. */
public final class Learner {
    public static final int FEATURES = 10, MIN_WEEKS = 80, TEST = 13, VALIDATION = 26, WINDOW = 156;
    public static final double BASELINE = -1, MOMENTUM = -2;
    private static final double[] CANDIDATES = {BASELINE, MOMENTUM, 0.1, 1, 10, 100};
    public static final String VERSION = "ridge-weekly-v1";

    public static double round(double v) { return Math.round(v * 10.0) / 10.0; }
    private static void check() { if (Thread.currentThread().isInterrupted()) throw new java.util.concurrent.CancellationException(); }
    public static double change(List<OilData.Row> r,int i,int f) { return r.get(i).prices[f]-r.get(i-1).prices[f]; }
    public static double[] features(List<OilData.Row> r,int t,int f) {
        if (t < 12 || t >= r.size()) throw new IllegalArgumentException("特徵視窗不足");
        double[] x = new double[FEATURES];
        for (int i=0;i<4;i++) x[i] = change(r,t-i,f);
        double m4=0,m12=0,level=0,var=0;
        for(int i=0;i<12;i++) { double d=change(r,t-i,f); m12+=d/12; level+=r.get(t-i).prices[f]/12; if(i<4)m4+=d/4; }
        for(int i=0;i<12;i++) {double d=change(r,t-i,f)-m12; var+=d*d/12;}
        x[4]=m4; x[5]=m12; x[6]=Math.sqrt(var); x[7]=r.get(t).prices[f]-level;
        x[8]=change(r,t,f==3 ? 1 : 3); x[9]=r.get(t).prices[f]/30.0;
        return x;
    }

    public static final class Fit {
        public double kind, intercept;
        public double[] mean=new double[FEATURES], scale=new double[FEATURES], weights=new double[FEATURES];
        public double predict(List<OilData.Row> rows,int t,int f) {
            double delta=0;
            if(kind==MOMENTUM) for(int i=0;i<4;i++) delta+=0.5*change(rows,t-i,f)/4;
            else if(kind>=0) {
                double[] x=features(rows,t,f); delta=intercept;
                for(int j=0;j<FEATURES;j++) delta+=weights[j]*(x[j]-mean[j])/scale[j];
            }
            if(!Double.isFinite(delta)) throw new IllegalArgumentException("模型計算未收斂");
            return round(Math.max(5,Math.min(100,rows.get(t).prices[f]+Math.max(-5,Math.min(5,delta)))));
        }
        public JSONObject json() throws JSONException {
            return new JSONObject().put("kind",kind).put("intercept",intercept).put("mean",array(mean)).put("scale",array(scale)).put("weights",array(weights));
        }
        public static Fit from(JSONObject j) throws JSONException {
            Fit f=new Fit();f.kind=j.getDouble("kind");f.intercept=j.getDouble("intercept");
            f.mean=vector(j.getJSONArray("mean"));f.scale=vector(j.getJSONArray("scale"));f.weights=vector(j.getJSONArray("weights"));return f;
        }
    }

    public static Fit fit(List<OilData.Row> rows,int origin,int f,double kind) {
        check();Fit model=new Fit();model.kind=kind;Arrays.fill(model.scale,1);
        if(kind<0)return model;
        int start=Math.max(12,origin-WINDOW), n=origin-start;
        if(n<20)throw new IllegalArgumentException("訓練樣本不足");
        double[][] x=new double[n][];double[] y=new double[n];
        for(int i=0;i<n;i++) {
            int t=start+i;x[i]=features(rows,t,f);y[i]=change(rows,t+1,f);model.intercept+=y[i]/n;
            for(int j=0;j<FEATURES;j++)model.mean[j]+=x[i][j]/n;
        }
        for(int j=0;j<FEATURES;j++){
            double v=0;for(int i=0;i<n;i++){double d=x[i][j]-model.mean[j];v+=d*d/n;}
            model.scale[j]=Math.max(1e-6,Math.sqrt(v));
        }
        double[][] a=new double[FEATURES][FEATURES];double[] b=new double[FEATURES];
        for(int i=0;i<n;i++){
            for(int j=0;j<FEATURES;j++)x[i][j]=(x[i][j]-model.mean[j])/model.scale[j];
            for(int j=0;j<FEATURES;j++){
                b[j]+=x[i][j]*(y[i]-model.intercept);
                for(int k=0;k<FEATURES;k++)a[j][k]+=x[i][j]*x[i][k];
            }
        }
        for(int j=0;j<FEATURES;j++)a[j][j]+=kind;
        model.weights=solve(a,b);return model;
    }

    private static double[] solve(double[][] a,double[] b){
        int n=b.length;
        for(int c=0;c<n;c++){
            int pivot=c;for(int i=c+1;i<n;i++)if(Math.abs(a[i][c])>Math.abs(a[pivot][c]))pivot=i;
            double[] tmp=a[c];a[c]=a[pivot];a[pivot]=tmp;double v=b[c];b[c]=b[pivot];b[pivot]=v;
            if(Math.abs(a[c][c])<1e-12)throw new IllegalArgumentException("矩陣不可解");
            v=a[c][c];for(int k=c;k<n;k++)a[c][k]/=v;b[c]/=v;
            for(int i=0;i<n;i++)if(i!=c){v=a[i][c];for(int k=c;k<n;k++)a[i][k]-=v*a[c][k];b[i]-=v*b[c];}
        }
        return b;
    }

    public static final class Score {
        public double mae, direction, movementRecall, falseMoveRate;
        public int movingCount, movementHits;
        public double[] errors;
        public JSONArray points=new JSONArray();
    }
    public static Score evaluate(List<OilData.Row> rows,int fuel,double kind,int from,int to)throws JSONException{
        Score s=new Score();s.errors=new double[to-from];int correct=0;
        for(int t=from;t<to;t++){
            check();double pred=fit(rows,t-1,fuel,kind).predict(rows,t-1,fuel), actual=rows.get(t).prices[fuel],last=rows.get(t-1).prices[fuel];
            double error=Math.abs(pred-actual);s.errors[t-from]=error;s.mae+=error/(to-from);
            if(direction(pred-last)==direction(actual-last))correct++;
            s.points.put(new JSONObject().put("date",rows.get(t).date.toString()).put("actual",actual).put("predicted",pred));
        }
        s.direction=100.0*correct/(to-from);return s;
    }
    private static int direction(double x){return x>0.049?1:x<-.049?-1:0;}
    public static double quantile(double[] v,double q){double[] a=v.clone();Arrays.sort(a);return a[Math.max(0,Math.min(a.length-1,(int)Math.ceil((a.length+1)*q)-1))];}

    public static JSONObject train(List<OilData.Row> rows,int revision,String hash)throws Exception{
        if(rows.size()<MIN_WEEKS)throw new IllegalArgumentException("至少需要 "+MIN_WEEKS+" 週資料才能訓練");
        int n=rows.size(),testStart=n-TEST,valStart=testStart-VALIDATION;
        JSONArray fuels=new JSONArray();
        for(int f=0;f<4;f++){
            double best=BASELINE,bestScore=Double.POSITIVE_INFINITY,ridge=1,ridgeScore=Double.POSITIVE_INFINITY;
            JSONArray scores=new JSONArray();
            for(double k:CANDIDATES){
                Score s=evaluate(rows,f,k,valStart,testStart);
                scores.put(new JSONObject().put("kind",k).put("mae",s.mae));
                if(s.mae<bestScore-0.005){best=k;bestScore=s.mae;}
                if(k>=0&&s.mae<ridgeScore){ridge=k;ridgeScore=s.mae;}
            }
            Score baseline=evaluate(rows,f,BASELINE,testStart,n),candidate=evaluate(rows,f,best,testStart,n);
            // This 13-week segment is a promotion gate, not an unbiased live-performance estimate.
            double active=best!=BASELINE&&candidate.mae+0.01<baseline.mae ? best : BASELINE;
            Score score=active==BASELINE?baseline:candidate;
            Score calibration=evaluate(rows,f,active,Math.max(40,n-26),n);
            Fit finalFit=fit(rows,n-1,f,active), trainedRidge=fit(rows,n-1,f,ridge);
            double predicted=finalFit.predict(rows,n-1,f);
            double radius=Math.max(0.1,Math.ceil(quantile(calibration.errors,0.8)*10)/10);
            fuels.put(new JSONObject().put("fuel",f).put("fit",finalFit.json()).put("candidateRidge",trainedRidge.json())
                .put("kind",active).put("name",name(active)).put("candidateKind",best).put("candidateName",name(best))
                .put("mae",score.mae).put("baselineMae",baseline.mae).put("candidateMae",candidate.mae)
                .put("direction",score.direction).put("testCount",TEST).put("validationCount",VALIDATION)
                .put("trainingSamples",Math.min(WINDOW,n-13)).put("predicted",predicted)
                .put("lower",Math.max(5,round(predicted-radius))).put("upper",Math.min(100,round(predicted+radius)))
                .put("radius",radius).put("validationScores",scores).put("backtest",score.points)
                .put("decision",active==BASELINE?"候選模型未明顯勝過價格不變基準，本輪採用基準。":"候選模型通過最近 13 週驗收，已重新訓練並啟用。"));
        }
        return new JSONObject().put("schema",1).put("algorithm",VERSION).put("revision",revision).put("dataHash",hash)
            .put("trainedAt",java.time.Instant.now().toString()).put("trainedThrough",rows.get(n-1).date.toString())
            .put("targetDate",rows.get(n-1).date.plusWeeks(1).toString()).put("weeks",n)
            .put("testStart",rows.get(testStart).date.toString()).put("testEnd",rows.get(n-1).date.toString()).put("fuels",fuels);
    }
    public static String name(double kind){return kind==BASELINE?"價格不變基準":kind==MOMENTUM?"短期趨勢模型":"嶺迴歸模型";}
    public static JSONArray array(double[] v)throws JSONException{JSONArray a=new JSONArray();for(double x:v)a.put(x);return a;}
    private static double[] vector(JSONArray a)throws JSONException{if(a.length()!=FEATURES)throw new IllegalArgumentException("模型維度錯誤");double[] v=new double[FEATURES];for(int i=0;i<v.length;i++)v[i]=a.getDouble(i);return v;}
}
