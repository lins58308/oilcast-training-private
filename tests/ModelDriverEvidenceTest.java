import java.nio.file.*;
import java.time.*;
import java.util.*;
import org.json.*;
import tw.oilcast.core.*;

/** Verify that historical event/week outcomes are truly paired, not merely listed. */
public final class ModelDriverEvidenceTest {
    private static int count;
    private static void ok(boolean b,String what){count++;if(!b)throw new AssertionError(what);}
    private static JSONObject news(String published,String available)throws Exception {
        return ContextData.article("Oil production supply cut",
          "https://www.eia.gov/todayinenergy/impacttest","EIA",
          Instant.parse(published),Instant.parse(available),true);
    }
    public static void main(String[] args)throws Exception{
        Path root=Path.of(args[0]);
        List<OilData.Row> history=OilData.weeklyThrough(
            OilData.fromJson(new JSONObject(Files.readString(root.resolve("data/seed.json"))).getJSONArray("rows")),
            LocalDate.parse("2026-09-07"));
        JSONObject raw=ContextData.empty();
        // Article is known at Monday 2020-05-04 origin and is paired with
        // the following Monday's CPC move, not the same week's observed change.
        raw.getJSONArray("news").put(news("2020-05-03T02:00:00Z","2020-05-03T03:00:00Z"));
        MultiLearner.Data d=new MultiLearner.Data(history,new ContextData(raw),Instant.parse("2026-09-13T13:00:00Z"));
        JSONObject analysis=WeeklyDriverEvidence.analyze(d,0);
        ok(analysis.getString("supervisedTarget").equals("next-week-cpc-effective-price-delta"),"target must be next-week CPC outcome");
        JSONArray factors=analysis.getJSONArray("associations");
        ok(factors.length()==9,"exact set of crude, FX and six news topics is checked");
        for(int i=0;i<factors.length();i++){
            JSONObject row=factors.getJSONObject(i);
            ok(row.getInt("observedWeeks")>=row.getInt("signalWeeks"),"signal count is within measured history");
            ok(row.getInt("changedWeeks")<=row.getInt("observedWeeks"),"changing weeks cannot exceed historical origins");
            if(!row.getBoolean("ready"))ok(row.getString("causality").contains("不能單獨判定因果"),"insufficient evidence is not causal proof");
        }
        JSONObject supply=factors.getJSONObject(3); // feature 18, supply cuts
        ok(supply.getInt("featureIndex")==18&&supply.getInt("signalWeeks")>=1&&supply.getInt("signalWeeks")<=4,"the story influences only its subsequent 28-day Monday origins");
        ok(supply.getInt("observedWeeks")==supply.getInt("signalWeeks"),"weeks without news are missing, not negative news evidence");
        ok(!supply.getBoolean("ready"),"one news item cannot yield a reliable historical association");
        MultiLearner.Data futureLeak=new MultiLearner.Data(history,new ContextData(
            ContextData.empty().put("news",new JSONArray().put(news("2020-05-03T02:00:00Z","2026-10-01T00:00:00Z")))),
            Instant.parse("2026-09-13T13:00:00Z"));
        ok(WeeklyDriverEvidence.analyze(futureLeak,0).getJSONArray("associations")
           .getJSONObject(3).getInt("signalWeeks")==0,"later-discovered news cannot affect an earlier training origin");
        System.out.println("PASS: "+count+" supervised news/crude weekly driver association checks");
    }
}
