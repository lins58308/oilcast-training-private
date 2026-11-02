import java.nio.file.*;
import java.time.*;
import java.util.*;
import org.json.*;
import tw.oilcast.core.*;

/**
 * The 13-week audit cannot choose v6/v7 families or hyperparameters.
 * Mutate all targets in this final holdout and verify that only audit
 * diagnostics and final re-fitted weights change; earlier validation
 * decisions must remain identical.
 */
public final class PromotionLeakageTest {
    private static int checks=0;
    private static void ok(boolean good,String reason) {
        checks++;
        if(!good)throw new AssertionError(reason);
    }
    public static void main(String[] args)throws Exception {
        JSONObject seed=new JSONObject(Files.readString(Path.of("data/seed.json")));
        List<OilData.Row> original=OilData.weeklyThrough(
           OilData.fromJson(seed.getJSONArray("rows")),LocalDate.parse("2026-09-07"));
        List<OilData.Row> modified=new ArrayList<>(original);
        for(int i=modified.size()-13;i<modified.size();i++){
            OilData.Row row=modified.get(i);
            double[] prices=row.prices.clone();
            for(int f=0;f<4;f++)prices[f]-=(i%2==0?0.6:0.4);
            modified.set(i,new OilData.Row(row.date,prices));
        }
        ContextData context=new ContextData(new JSONObject(
            Files.readString(Path.of("data/multisource.json"))));
        Instant now=Instant.parse("2026-09-13T13:00:00Z");
        JSONObject initial=MultiLearner.train(original,context,1,"unmodified",now);
        JSONObject changed=MultiLearner.train(modified,context,2,"mutated",now);
        JSONArray before=initial.getJSONArray("fuels"),after=changed.getJSONArray("fuels");
        int alteredAudit=0;
        for(int f=0;f<4;f++){
            JSONObject a=before.getJSONObject(f),b=after.getJSONObject(f);
            ok(a.getInt("family")==b.getInt("family"),
               "holdout changes may not affect model selection for fuel "+f);
            JSONArray va=a.getJSONArray("comparisons"),vb=b.getJSONArray("comparisons");
            for(int family=0;family<va.length();family++){
                JSONObject ca=va.getJSONObject(family),cb=vb.getJSONObject(family);
                ok(Math.abs(ca.getDouble("validationMae")-cb.getDouble("validationMae"))<1e-9,
                   "holdout changes must not affect pre-audit validation");
            }
            if(Math.abs(a.getDouble("mae")-b.getDouble("mae"))>.0001)alteredAudit++;
        }
        ok(alteredAudit>0,"locked audit must still detect intentionally altered test outcomes");
        System.out.println("PASS: "+checks+" chronological no-holdout-selection tests");
    }
}
