import java.nio.file.*;
import java.time.*;
import java.util.*;
import org.json.*;
import tw.oilcast.core.*;

/** Compare inference-compatible candidate heads on frozen input histories.
 * Each block's fit is fixed BEFORE that block; all feature origins stay causal.
 * These are candidate diagnostics, not a claim that the active model passed.
 */
public final class SignedCoreMaeAudit {
    private static JSONObject score(MultiLearner.Data d,int fuel,int family,double regularizer,int start,int end)throws Exception{
        MultiLearner.Fit fit=MultiLearner.fit(d,start-1,fuel,family,regularizer);
        double mae=0,baseline=0,rmse=0;int changed=0,hits=0,falseMoves=0,flat=0;
        for(int t=start;t<end;t++){
            double base=d.rows.get(t-1).prices[fuel],actual=d.rows.get(t).prices[fuel];
            double predicted=fit.price(base,d.x[fuel][t-1]);
            double error=Math.abs(actual-predicted);mae+=error;rmse+=error*error;baseline+=Math.abs(actual-base);
            int a=direction(actual-base),p=direction(predicted-base);
            if(a!=0){changed++;if(a==p)hits++;}else{flat++;if(p!=0)falseMoves++;}
        }
        return new JSONObject().put("mae",mae/(end-start)).put("baselineMae",baseline/(end-start))
            .put("rmse",Math.sqrt(rmse/(end-start))).put("weeks",end-start).put("changedWeeks",changed)
            .put("changedWeekRecall",changed==0?JSONObject.NULL:(double)hits/changed)
            .put("falseMoveRate",flat==0?JSONObject.NULL:(double)falseMoves/flat)
            .put("samples",fit.samples);
    }
    private static int direction(double d){return d>.049?1:d<-.049?-1:0;}
    public static void main(String[] args)throws Exception{
        if(args.length!=3)throw new IllegalArgumentException("checkpoint output asOf");
        JSONObject state=new JSONObject(Files.readString(Path.of(args[0])));Instant now=Instant.parse(args[2]);
        List<OilData.Row> rows=OilData.weeklyThrough(OilData.fromJson(state.getJSONArray("rows")),OilData.monday(now.atZone(OilData.TAIPEI).toLocalDate()));
        MultiLearner.Data data=new MultiLearner.Data(rows,new ContextData(state.getJSONObject("multisource")),now);
        int n=rows.size(),audit=n-13;JSONArray periods=new JSONArray();
        List<int[]> ranges=new ArrayList<>();
        for(int gap:new int[]{260,208,156,104,52})ranges.add(new int[]{audit-gap-13,audit-gap});
        ranges.add(new int[]{audit-78,audit-39});ranges.add(new int[]{audit-39,audit});ranges.add(new int[]{audit,n});
        int[] families=MultiLearner.FAMILY.length>6?new int[]{4,6}:new int[]{4};
        for(int index=0;index<ranges.size();index++){
            int start=ranges.get(index)[0],end=ranges.get(index)[1];JSONArray heads=new JSONArray();
            for(int family:families){
                JSONArray fuels=new JSONArray();
                for(int f=0;f<4;f++)fuels.put(score(data,f,family,family==6?.01:3.,start,end).put("fuel",f));
                heads.put(new JSONObject().put("family",family).put("regularizer",family==6?.01:3.).put("fuels",fuels));
            }
            periods.put(new JSONObject().put("role",index<5?"earlier-fold":index<7?"recent-validation":"previously-inspected-audit")
                .put("start",rows.get(start).date.toString()).put("end",rows.get(end-1).date.toString()).put("heads",heads));
            System.out.println("Signed-core frozen block complete: "+rows.get(start).date+" to "+rows.get(end-1).date);
        }
        JSONObject result=new JSONObject().put("algorithm",MultiLearner.VERSION).put("weeks",n)
            .put("asOf",now.toString()).put("periods",periods).put("candidateDiagnosticsOnly",true)
            .put("fitFrozenBeforeEachBlock",true).put("holdoutUsedForSelection",false).put("automaticPromotionApproved",false);
        Path out=Path.of(args[1]);Files.createDirectories(out.toAbsolutePath().getParent());Files.writeString(out,result.toString(2));
    }
}
