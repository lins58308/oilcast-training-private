package tw.oilcast.core;

import org.json.*;
import java.time.*;
import java.util.*;

/** Ridge and compact deep neural candidates, assessed with chronological walk-forward backtests. */
public final class MultiLearner {
    public static final String VERSION="multisource-neural-v7";
    public static boolean supports(String algorithm){return VERSION.equals(algorithm)||"multisource-adaptive-v6".equals(algorithm);}
    public static final String[] FAMILY={"價格不變基準","歷史牌價模型","牌價＋原油匯率模型","牌價＋行情＋新聞模型","近期加權穩健模型","三模型平均集成","雙隱藏層深度學習模型"};
    private static final double[] LAMBDAS={0.3,3,30,300};
    private static final double[] NEURAL_DECAYS={.001,.01};
    private static final int P=ContextData.DIM;
    private static final int WORK_PER_FUEL=40+(FAMILY.length-1)*118+26*FAMILY.length;
    private static void check(){if(Thread.currentThread().isInterrupted())throw new java.util.concurrent.CancellationException();}
    public static final class Data {
        public final List<OilData.Row> rows;public final double[][][] x;public final double[][] live;
        public final ContextData.Snapshot snapshot;public int newsWeeks,marketWeeks;
        private final Map<String,Fit> fits=new HashMap<>();
        public Data(List<OilData.Row> rows,ContextData context,Instant now)throws Exception{
            this(rows,context,now,ProgressListener.NONE);
        }
        public Data(List<OilData.Row> rows,ContextData context,Instant now,ProgressListener progress)throws Exception{
            this.rows=rows;x=new double[4][rows.size()][P];live=new double[4][P];snapshot=context.snapshot(now);
            for(int t=12;t<rows.size();t++){
                check();progress.update("features",t-12,rows.size()-12,-1,-1);
                Instant cutoff=rows.get(t).date.atStartOfDay(OilData.TAIPEI).toInstant();if(cutoff.isAfter(now))cutoff=now;
                double[] outside=context.snapshot(cutoff).values;
                if(outside[17]>.99)marketWeeks++;if(outside[25]>.99)newsWeeks++;
                for(int f=0;f<4;f++){x[f][t]=outside.clone();System.arraycopy(Learner.features(rows,t,f),0,x[f][t],0,10);}
            }
            for(int f=0;f<4;f++){live[f]=snapshot.values.clone();System.arraycopy(Learner.features(rows,rows.size()-1,f),0,live[f],0,10);}
        }
    }
    public static final class Fit {
        public int family;public double lambda,intercept,clip;public int samples;
        public NeuralLearner.Network neural;
        public double[] mean=new double[P],scale=new double[P],weights=new double[P];
        public Fit(){Arrays.fill(scale,1);}
        private double standard(double input,int i){double z=(input-mean[i])/scale[i];return clip>0?Math.max(-clip,Math.min(clip,z)):z;}
        public double[] terms(double[] input){
            double[] a=new double[P];
            if(neural==null){for(int i=0;i<P;i++)a[i]=weights[i]*standard(input[i],i);return a;}
            double[] z=new double[P];for(int i=0;i<P;i++)z[i]=standard(input[i],i);
            double full=neural.predict(z),base=neural.predict(new double[P]),sum=0;
            for(int i=0;i<P;i++){double prior=z[i];z[i]=0;a[i]=full-neural.predict(z);z[i]=prior;sum+=a[i];}
            if(Math.abs(sum)>1e-9)for(int i=0;i<P;i++)a[i]*=(full-base)/sum;
            else Arrays.fill(a,0);
            return a;
        }
        public double delta(double[] input){
            double d;
            if(neural!=null){double[] z=new double[P];for(int i=0;i<P;i++)z[i]=standard(input[i],i);d=neural.predict(z);}
            else{d=intercept;for(double a:terms(input))d+=a;}
            if(!Double.isFinite(d))throw new IllegalArgumentException("多來源模型計算失敗");return d;
        }
        public double price(double base,double[] input){return Learner.round(Math.max(5,Math.min(100,base+Math.max(-5,Math.min(5,delta(input))))));}
        public JSONObject json()throws JSONException{JSONObject result=new JSONObject().put("family",family).put("lambda",lambda).put("clip",clip).put("samples",samples).put("intercept",intercept).put("mean",Learner.array(mean)).put("scale",Learner.array(scale)).put("weights",Learner.array(weights));if(neural!=null)result.put("neural",neural.json());return result;}
        public static Fit from(JSONObject j)throws JSONException{
            Fit m=new Fit();m.family=j.getInt("family");m.lambda=j.getDouble("lambda");m.samples=j.getInt("samples");m.intercept=j.getDouble("intercept");m.clip=j.optDouble("clip",0);
            if(m.family<0||m.family>=FAMILY.length||!Double.isFinite(m.intercept)||!Double.isFinite(m.clip)||m.clip<0||m.clip>10)throw new IllegalArgumentException("模型參數異常");
            JSONArray a=j.getJSONArray("mean"),b=j.getJSONArray("scale"),c=j.getJSONArray("weights");
            if(a.length()!=P||b.length()!=P||c.length()!=P)throw new IllegalArgumentException("模型維度錯誤");
            for(int i=0;i<P;i++){m.mean[i]=a.getDouble(i);m.scale[i]=b.getDouble(i);m.weights[i]=c.getDouble(i);if(!Double.isFinite(m.weights[i])||!Double.isFinite(m.mean[i])||!Double.isFinite(m.scale[i])||m.scale[i]<=0)throw new IllegalArgumentException("模型參數異常");}
            if(m.family==6)m.neural=NeuralLearner.Network.from(j.getJSONObject("neural"));
            else if(j.has("neural"))throw new IllegalArgumentException("非神經網路模型含異常參數");return m;
        }
        public JSONObject explain(double base,double[] input,String cutoff)throws JSONException{
            double[] contribution=terms(input),groups=new double[ContextData.GROUPS.length];JSONArray terms=new JSONArray(),g=new JSONArray();
            for(int i=0;i<P;i++){
                groups[ContextData.group(i)]+=contribution[i];
                double standardized=standard(input[i],i);
                terms.put(new JSONObject().put("name",ContextData.NAMES[i]).put("value",input[i]).put("standardized",standardized).put("unit",ContextData.unit(i)).put("mean",mean[i]).put("coefficient",neural==null?weights[i]:(Math.abs(standardized)<1e-9?0:contribution[i]/standardized)).put("scale",scale[i]).put("contribution",contribution[i]));
            }
            for(int i=0;i<groups.length;i++)g.put(new JSONObject().put("name",ContextData.GROUPS[i]).put("contribution",groups[i]));
            double raw=delta(input),bounded=Math.max(5,Math.min(100,base+Math.max(-5,Math.min(5,raw)))),finalPrice=price(base,input);
            double localIntercept=raw;for(double item:contribution)localIntercept-=item;
            return new JSONObject().put("family",family).put("name",FAMILY[family]).put("basePrice",base).put("intercept",localIntercept).put("rawDelta",raw)
                .put("limitAdjustment",bounded-base-raw).put("roundingAdjustment",finalPrice-bounded).put("predicted",finalPrice)
                .put("groups",g).put("terms",terms).put("cutoff",cutoff)
                .put("note",family==0?"目前採用價格不變基準，可在內部分析診斷查看候選的回測比較。":neural!=null?"雙隱藏層非線性神經網路。特徵貢獻為對均值輸入做逐項移除後正規化的局部近似歸因，並非因果解釋。":"各項是相對訓練均值的數學分解，不代表已證實的因果關係。");
        }
    }
    private static boolean used(int i,int family){return family>=3||family==2&&i<18||family==1&&i<10;}
    public static Fit fit(Data d,int origin,int fuel,int family,double lambda){
        check();if(family<0||family>=FAMILY.length)throw new IllegalArgumentException("未知模型");
        String key=origin+":"+fuel+":"+family+":"+lambda;Fit cached=d.fits.get(key);if(cached!=null)return cached;
        Fit m=new Fit();m.family=family;m.lambda=lambda;if(family==0)return m;
        if(family==5){
            Fit[] parts={fit(d,origin,fuel,1,lambda),fit(d,origin,fuel,2,lambda),fit(d,origin,fuel,3,lambda)};
            m.mean=parts[2].mean.clone();m.scale=parts[2].scale.clone();m.samples=parts[2].samples;
            for(Fit part:parts){m.intercept+=part.intercept/3;
                for(int j=0;j<P;j++){m.weights[j]+=part.weights[j]*m.scale[j]/part.scale[j]/3;m.intercept+=part.weights[j]*(m.mean[j]-part.mean[j])/part.scale[j]/3;}}
            d.fits.put(key,m);return m;
        }
        int start=Math.max(12,origin-156),n=origin-start;m.samples=n;if(n<24)throw new IllegalArgumentException("多來源訓練樣本不足");
        if(family==6){
            m.clip=4;
            double[][] x=new double[n][P];double[] y=new double[n];double sum=0;
            for(int i=0;i<n;i++){
                double w=Math.pow(.5,(n-1-i)/26.0);sum+=w;y[i]=Learner.change(d.rows,start+i+1,fuel);
                for(int j=0;j<P;j++){x[i][j]=d.x[fuel][start+i][j];m.mean[j]+=w*x[i][j];}
            }
            for(int j=0;j<P;j++)m.mean[j]/=sum;
            for(int j=0;j<P;j++){
                double variance=0;
                for(int i=0;i<n;i++)variance+=Math.pow(.5,(n-1-i)/26.0)*Math.pow(x[i][j]-m.mean[j],2)/sum;
                m.scale[j]=Math.max(1e-5,Math.sqrt(variance));
                for(int i=0;i<n;i++)x[i][j]=m.standard(x[i][j],j);
            }
            m.neural=NeuralLearner.fit(x,y,lambda,0x5deece1L+31L*fuel);
            d.fits.put(key,m);return m;
        }
        m.clip=family==4?4:0;
        double[][] x=new double[n][P];double[] y=new double[n],sampleWeights=new double[n],zmean=new double[P];double sum=0;
        for(int i=0;i<n;i++){sampleWeights[i]=family==4?Math.pow(.5,(n-1-i)/26.0):1;sum+=sampleWeights[i];}
        for(int i=0;i<n;i++)sampleWeights[i]*=n/sum;
        for(int i=0;i<n;i++){double w=sampleWeights[i]/n;y[i]=Learner.change(d.rows,start+i+1,fuel);m.intercept+=w*y[i];
            for(int j=0;j<P;j++)if(used(j,family)){x[i][j]=d.x[fuel][start+i][j];m.mean[j]+=w*x[i][j];}}
        for(int j=0;j<P;j++){double v=0;for(int i=0;i<n;i++)v+=sampleWeights[i]*Math.pow(x[i][j]-m.mean[j],2)/n;m.scale[j]=v<1e-12?1:Math.sqrt(v);}
        double[][] a=new double[P][P];double[] b=new double[P];
        for(int i=0;i<n;i++)for(int j=0;j<P;j++){x[i][j]=m.standard(x[i][j],j);zmean[j]+=sampleWeights[i]*x[i][j]/n;}
        for(int i=0;i<n;i++){
            for(int j=0;j<P;j++){double z=x[i][j]-zmean[j];b[j]+=sampleWeights[i]*z*(y[i]-m.intercept);
                for(int k=0;k<P;k++)a[j][k]+=sampleWeights[i]*z*(x[i][k]-zmean[k]);}
        }
        for(int i=0;i<P;i++)a[i][i]+=lambda;
        for(int c=0;c<P;c++){
            int p=c;for(int i=c+1;i<P;i++)if(Math.abs(a[i][c])>Math.abs(a[p][c]))p=i;
            double[] row=a[c];a[c]=a[p];a[p]=row;double v=b[c];b[c]=b[p];b[p]=v;
            if(Math.abs(a[c][c])<1e-12)throw new IllegalArgumentException("多來源模型矩陣不可解");v=a[c][c];for(int k=c;k<P;k++)a[c][k]/=v;b[c]/=v;
            for(int i=0;i<P;i++)if(i!=c){v=a[i][c];for(int k=c;k<P;k++)a[i][k]-=v*a[c][k];b[i]-=v*b[c];}
        }m.weights=b;for(int j=0;j<P;j++)m.intercept-=b[j]*zmean[j];d.fits.put(key,m);return m;
    }
    public static Learner.Score evaluate(Data d,int f,int family,double lambda,int start,int end)throws Exception{
        return evaluate(d,f,family,lambda,start,end,ProgressListener.NONE,new int[]{0});
    }
    private static Learner.Score evaluate(Data d,int f,int family,double lambda,int start,int end,ProgressListener progress,int[] done)throws Exception{
        Learner.Score s=new Learner.Score();s.errors=new double[end-start];int correct=0,moving=0,captured=0,falseMoves=0;
        for(int t=start;t<end;t++){
            Fit fit=fit(d,t-1,f,family,lambda);double base=d.rows.get(t-1).prices[f],actual=d.rows.get(t).prices[f],pred=fit.price(base,d.x[f][t-1]);
            progress.update("fit",++done[0],4*WORK_PER_FUEL,f,family);
            double error=Math.abs(pred-actual);s.errors[t-start]=error;s.mae+=error/(end-start);
            int actualDirection=dir(actual-base),predictedDirection=dir(pred-base);
            if(predictedDirection==actualDirection)correct++;
            if(actualDirection!=0){moving++;if(predictedDirection==actualDirection)captured++;}
            else if(predictedDirection!=0)falseMoves++;
            s.points.put(new JSONObject().put("date",d.rows.get(t).date.toString()).put("actual",actual).put("predicted",pred).put("base",base));
        }
        s.direction=100.0*correct/(end-start);
        s.movementRecall=moving==0?0:100.0*captured/moving;
        s.falseMoveRate=end-start-moving==0?0:100.0*falseMoves/(end-start-moving);
        s.movingCount=moving;s.movementHits=captured;
        return s;
    }
    private static int dir(double d){return d>.049?1:d<-.049?-1:0;}
    /** Price errors remain primary; rewarding correctly captured moves avoids the always-flat trap. */
    private static double objective(Learner.Score s){return s.mae-.11*s.movementRecall/100-.02*s.direction/100+.03*s.falseMoveRate/100;}
    public static JSONObject train(List<OilData.Row> rows,ContextData context,int revision,String hash,Instant now)throws Exception{
        return train(rows,context,revision,hash,now,ProgressListener.NONE);
    }
    public static JSONObject train(List<OilData.Row> rows,ContextData context,int revision,String hash,Instant now,ProgressListener progress)throws Exception{
        if(rows.size()<80)throw new IllegalArgumentException("至少需要 80 週歷史");
        Data d=new Data(rows,context,now,progress);int[] done={0};int n=rows.size(),test=n-13,val=Math.max(40,test-39);JSONArray fuels=new JSONArray();
        boolean marketOK=d.marketWeeks>=80&&d.snapshot.values[17]>.99,newsOK=marketOK&&d.newsWeeks>=80&&d.snapshot.values[25]>.99;
        boolean[] eligible={true,true,marketOK,newsOK,newsOK,newsOK,true};
        for(int f=0;f<4;f++){
            JSONArray comparisons=new JSONArray();JSONArray familyList=new JSONArray();
            double[] bestLambda=new double[FAMILY.length],validation=new double[FAMILY.length],validationObjective=new double[FAMILY.length];Learner.Score[] gate=new Learner.Score[FAMILY.length],valScore=new Learner.Score[FAMILY.length];
            Fit[] allFits=new Fit[FAMILY.length];
            int candidate=0;valScore[0]=evaluate(d,f,0,0,val,test,progress,done);validation[0]=valScore[0].mae;
            validationObjective[0]=objective(valScore[0]);gate[0]=evaluate(d,f,0,0,test,n,progress,done);
            allFits[0]=fit(d,n-1,f,0,0);progress.update("fit",++done[0],4*WORK_PER_FUEL,f,0);
            comparisons.put(new JSONObject().put("family",0).put("name",FAMILY[0]).put("validationMae",validation[0]).put("validationObjective",validationObjective[0]).put("mae",gate[0].mae).put("eligible",true));
            for(int family=1;family<FAMILY.length;family++){
                validationObjective[family]=Double.POSITIVE_INFINITY;
                for(double lambda:family==6?NEURAL_DECAYS:LAMBDAS){
                    Learner.Score s=evaluate(d,f,family,lambda,val,test,progress,done);
                    if(objective(s)<validationObjective[family]){
                        validation[family]=s.mae;validationObjective[family]=objective(s);bestLambda[family]=lambda;valScore[family]=s;
                    }
                }
                gate[family]=evaluate(d,f,family,bestLambda[family],test,n,progress,done);
                allFits[family]=fit(d,n-1,f,family,bestLambda[family]);progress.update("fit",++done[0],4*WORK_PER_FUEL,f,family);
                comparisons.put(new JSONObject().put("family",family).put("name",FAMILY[family]).put("lambda",bestLambda[family]).put("validationMae",validation[family]).put("validationObjective",validationObjective[family])
                    .put("validationDirection",valScore[family].direction).put("validationMovementRecall",valScore[family].movementRecall).put("validationFalseMoveRate",valScore[family].falseMoveRate)
                    .put("mae",gate[family].mae).put("direction",gate[family].direction).put("movementRecall",gate[family].movementRecall).put("eligible",eligible[family]));
                if(eligible[family]&&validation[family]<=validation[0]+.045
                    &&(validationObjective[family]<validationObjective[candidate]-.001
                    ||candidate==0&&family==6&&valScore[family].movementRecall>=15
                        &&valScore[family].falseMoveRate<=55&&validation[family]<=validation[0]+.03))candidate=family;
            }
            // The latest 13 weeks are a locked out-of-sample audit, never a selection gate.
            int active=candidate;
            Fit fitted=allFits[active],all=allFits[3];
            double base=rows.get(n-1).prices[f],pred=fitted.price(base,d.live[f]);
            Learner.Score calibration=evaluate(d,f,active,bestLambda[active],n-26,n,progress,done);double radius=Math.max(.1,Math.ceil(Learner.quantile(calibration.errors,.8)*10)/10);
            String decision=active==0?"39 週滾動驗證中，候選未在限定誤差內顯示足夠的方向改善，保留價格不變基準。":"已依前 39 週滾動驗證選用「"+FAMILY[active]+"」，最近 13 週僅用作未參與選模的稽核。";
            
            for(int fam=0;fam<FAMILY.length;fam++){
                double famPred=allFits[fam].price(base,d.live[f]);
                Learner.Score familyErrors=fam==active?calibration:evaluate(d,f,fam,bestLambda[fam],n-26,n,progress,done);
                double famRadius=Math.max(.1,Math.ceil(Learner.quantile(familyErrors.errors,.8)*10)/10);
                familyList.put(new JSONObject()
                    .put("family",fam)
                    .put("name",FAMILY[fam])
                    .put("lambda",bestLambda[fam])
                    .put("mae",gate[fam].mae).put("direction",gate[fam].direction).put("movementRecall",gate[fam].movementRecall).put("falseMoveRate",gate[fam].falseMoveRate)
                    .put("movingWeeks",gate[fam].movingCount).put("movementHits",gate[fam].movementHits)
                    .put("baselineSkill",gate[0].mae>0?100*(1-gate[fam].mae/gate[0].mae):0).put("halfLifeWeeks",fam==4||fam==6?26:0)
                    .put("validationMae",validation[fam]).put("validationObjective",validationObjective[fam])
                    .put("predicted",famPred)
                    .put("delta",Learner.round(famPred-base))
                    .put("lower",Math.max(5,Learner.round(famPred-famRadius)))
                    .put("upper",Math.min(100,Learner.round(famPred+famRadius)))
                    .put("radius",famRadius)
                    .put("eligible",eligible[fam])
                    .put("isAuto",fam==active)
                    .put("backtest",gate[fam].points)
                    .put("explanation",allFits[fam].explain(base,d.live[f],now.toString())));
            }

            fuels.put(new JSONObject().put("fuel",f).put("family",active).put("kind",active==0?-1:bestLambda[active]).put("name",FAMILY[active])
                .put("fit",fitted.json()).put("candidateFit",all.json()).put("candidateEligible",eligible[3]).put("candidateName",FAMILY[3]).put("candidateKind",bestLambda[3])
                .put("mae",gate[active].mae).put("baselineMae",gate[0].mae).put("candidateMae",gate[3].mae).put("direction",gate[active].direction)
                .put("movementRecall",gate[active].movementRecall).put("falseMoveRate",gate[active].falseMoveRate)
                .put("testCount",13).put("validationCount",test-val).put("trainingSamples",Math.min(156,n-13)).put("predicted",pred)
                .put("lower",Math.max(5,Learner.round(pred-radius))).put("upper",Math.min(100,Learner.round(pred+radius))).put("radius",radius)
                .put("backtest",gate[active].points).put("comparisons",comparisons).put("decision",decision)
                .put("families",familyList)
                .put("explanation",fitted.explain(base,d.live[f],now.toString())).put("candidateExplanation",all.explain(base,d.live[f],now.toString())));
        }
        progress.update("fit",done[0],4*WORK_PER_FUEL,3,FAMILY.length-1);
        return new JSONObject().put("schema",2).put("algorithm",VERSION).put("revision",revision).put("dataHash",hash).put("trainedAt",now.toString())
            .put("trainedThrough",rows.get(n-1).date.toString()).put("targetDate",rows.get(n-1).date.plusWeeks(1).toString()).put("weeks",n)
            .put("testStart",rows.get(test).date.toString()).put("testEnd",rows.get(n-1).date.toString()).put("fuels",fuels).put("context",d.snapshot.json())
            .put("newsHistory",context.newsSize()).put("marketHistory",context.marketSize()).put("newsWeeks",d.newsWeeks).put("marketWeeks",d.marketWeeks)
            .put("timingNote","歷史回測在每週一截取資料；即時預測使用本次可取得資訊，週中更新的表現需另看安裝後追蹤。歷史行情採 10 日發布延遲估計，EIA 存檔報導自隔日可用；這些回溯快照不是原始發布版本。");
    }
}
