package tw.oilcast.core;

import java.util.*;
import org.json.*;

/**
 * Time-ordered association evidence, never a statement of causation.
 *
 * For each Monday origin t, only features already available at t are paired
 * with the CPC effective-price change from t to t+1. This diagnostic uses
 * the same Data.x feature snapshots and labels as the supervised regressors;
 * there are no future/target-week headlines in the feature vector.
 */
public final class WeeklyDriverEvidence {
    private WeeklyDriverEvidence() {}
    private static final int[] FACTORS={10,11,12,18,19,20,21,22,23};
    private static final int MIN_SIGNAL_WEEKS=8;
    private static double rounded(double x){return Math.round(x*10000.0)/10000.0;}
    private static JSONObject group(MultiLearner.Data d,int fuel,int feature) throws Exception {
        boolean news=feature>=18;
        int observations=0,withSignal=0,withoutSignal=0,changed=0;
        double sum=0,signalSum=0,noSignalSum=0,sx=0,sy=0,sxy=0,sx2=0,sy2=0;
        int first=-1,last=-1;
        for(int t=12;t<d.rows.size()-1;t++){
            double[] x=d.x[fuel][t];
            // "Unreported" is not "no event". Market data must have all
            // three feeds, news needs at least one actually visible story.
            if(news ? x[24]<=0 : x[17]<.999)continue;
            double signal=x[feature],actual=Learner.change(d.rows,t+1,fuel);
            if(!Double.isFinite(signal)||!Double.isFinite(actual))continue;
            if(first<0)first=t;
            last=t;
            observations++;sum+=actual;
            if(Math.abs(actual)>=.049)changed++;
            boolean active=news?signal>1e-9:Math.abs(signal)>=.1;
            if(active){withSignal++;signalSum+=actual;}
            else{withoutSignal++;noSignalSum+=actual;}
            sx+=signal;sy+=actual;sxy+=signal*actual;
            sx2+=signal*signal;sy2+=actual*actual;
        }
        JSONObject result=new JSONObject().put("featureIndex",feature)
            .put("signal",ContextData.NAMES[feature])
            .put("kind",news?"news-event":"market-change")
            .put("observedWeeks",observations).put("changedWeeks",changed)
            .put("signalWeeks",withSignal).put("otherWeeks",withoutSignal)
            .put("ready",withSignal>=MIN_SIGNAL_WEEKS&&withoutSignal>=MIN_SIGNAL_WEEKS)
            .put("firstOrigin",first<0?JSONObject.NULL:d.rows.get(first).date.toString())
            .put("lastOrigin",last<0?JSONObject.NULL:d.rows.get(last).date.toString())
            .put("label","下一週中油實際有效牌價變化，單位新台幣／公升")
            .put("causality","歷史統計關聯，不能單獨判定因果");
        result.put("meanChangeWithSignal",withSignal==0?JSONObject.NULL:rounded(signalSum/withSignal));
        result.put("meanChangeWithoutSignal",withoutSignal==0?JSONObject.NULL:rounded(noSignalSum/withoutSignal));
        result.put("meanDifference",withSignal==0||withoutSignal==0?JSONObject.NULL:
            rounded(signalSum/withSignal-noSignalSum/withoutSignal));
        double xx=observations*sx2-sx*sx,yy=observations*sy2-sy*sy;
        result.put("correlation",observations<MIN_SIGNAL_WEEKS||xx<=1e-12||yy<=1e-12?
            JSONObject.NULL:rounded((observations*sxy-sx*sy)/Math.sqrt(xx*yy)));
        return result;
    }
    public static JSONObject analyze(MultiLearner.Data d,int fuel)throws Exception{
        JSONArray associations=new JSONArray();
        for(int feature:FACTORS)associations.put(group(d,fuel,feature));
        return new JSONObject().put("schema",1).put("fuel",fuel)
            .put("supervisedTarget","next-week-cpc-effective-price-delta")
            .put("informationCutoff","origin-week-Monday-00:00-Asia/Taipei")
            .put("allWeeklyOrigins",Math.max(0,d.rows.size()-13))
            .put("associations",associations)
            .put("note","供需、地緣、庫存、原油與匯率事件和下一週調價的歷史統計對照，並非新聞因果證明或保證預測。");
    }
}
