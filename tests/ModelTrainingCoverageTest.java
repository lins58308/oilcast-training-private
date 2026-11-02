import java.nio.file.*;
import java.time.*;
import java.util.*;
import org.json.*;
import tw.oilcast.core.*;

/** All post-warmup CPC weeks must be part of every final expanding-window fit. */
public final class ModelTrainingCoverageTest {
    private static int assertions;
    private static void ok(boolean b,String msg){
        assertions++;
        if(!b)throw new AssertionError(msg);
    }
    private static double difference(double[] a,double[] b) {
        double sum=0;
        for(int i=0;i<a.length;i++)sum+=Math.abs(a[i]-b[i]);
        return sum;
    }
    private static boolean different(MultiLearner.Fit a,MultiLearner.Fit b) {
        return Math.abs(a.intercept-b.intercept)>1e-9
            || difference(a.weights,b.weights)>1e-7;
    }
    public static void main(String[] args)throws Exception {
        Path root=Path.of(args[0]);
        JSONArray source=new JSONObject(Files.readString(root.resolve("data/seed.json"))).getJSONArray("rows");
        List<OilData.Row> rows=OilData.weeklyThrough(OilData.fromJson(source),LocalDate.parse("2026-09-07"));
        int origin=rows.size()-1,expected=origin-12;
        ok(expected>156,"fixture must include data older than the historical 156-week limit");
        ok(rows.size()>320,"training spans all available CPC history");

        List<OilData.Row> changed=new ArrayList<>(rows);
        int oldIndex=30;  // Well before the old 156-week horizon.
        OilData.Row original=changed.get(oldIndex);
        double[] adjusted=original.prices.clone();adjusted[0]+=1.0;
        changed.set(oldIndex,new OilData.Row(original.date,adjusted));
        Learner.Fit ridge1=Learner.fit(rows,origin,0,1);
        Learner.Fit ridge2=Learner.fit(changed,origin,0,1);
        ok(difference(ridge1.weights,ridge2.weights)>1e-7,
            "historical ridge must react to a pre-156-week label");

        ContextData ctx=new ContextData(new JSONObject(Files.readString(root.resolve("data/multisource.json"))));
        Instant asOf=Instant.parse("2026-09-13T13:00:00Z");
        MultiLearner.Data base=new MultiLearner.Data(rows,ctx,asOf);
        MultiLearner.Data pastChanged=new MultiLearner.Data(changed,ctx,asOf);
        for(int family : new int[]{1,4}){
            MultiLearner.Fit full=MultiLearner.fit(base,origin,0,family,3.0);
            MultiLearner.Fit mutated=MultiLearner.fit(pastChanged,origin,0,family,3.0);
            ok(full.samples==expected,"all post-warmup weekly labels fit for family "+family);
            ok(different(full,mutated),"older-than-156 weeks change trained family "+family);
        }
        if(MultiLearner.VERSION.contains("neural")){
            MultiLearner.Fit full=MultiLearner.fit(base,origin,0,6,.001);
            MultiLearner.Fit mutated=MultiLearner.fit(pastChanged,origin,0,6,.001);
            ok(full.samples==expected,"neural model uses every post-warmup weekly label");
            ok(!full.json().toString().equals(mutated.json().toString()),
                "neural network training responds to genuinely old observations");
        }
        // The last observed week is a target of the penultimate origin; no look-ahead.
        double[] current=rows.get(origin).prices;
        ok(Double.isFinite(current[0]),"last observed week is complete");
        System.out.println("PASS: "+assertions+" v6/v7 expanding-history coverage assertions; sample pairs="+expected);
    }
}
