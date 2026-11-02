import java.nio.file.*;
import java.time.*;
import java.util.*;
import org.json.*;
import tw.oilcast.core.*;

public final class SemanticResearchDatasetTest {
    private static int checks;
    private static void ok(boolean condition,String explanation){
        checks++;if(!condition)throw new AssertionError(explanation);
    }
    public static void main(String[] args)throws Exception{
        Path root=Path.of(args[0]);
        JSONArray official=new JSONObject(Files.readString(root.resolve("data/seed.json"))).getJSONArray("rows");
        JSONObject market=new JSONObject(Files.readString(root.resolve("data/multisource.json")));
        JSONObject late=ContextData.article("War disrupts Suez shipping",
            "https://www.bbc.com/news/world-late-record","BBC",
            Instant.parse("2020-05-01T00:00:00Z"),
            Instant.parse("2026-10-30T00:00:00Z"),false);
        JSONObject early=ContextData.article("Conflict shuts Suez shipping",
            "https://www.bbc.com/news/world-early-record","BBC",
            Instant.parse("2020-05-01T00:00:00Z"),
            Instant.parse("2020-05-02T00:00:00Z"),false);
        market.getJSONArray("news").put(late).put(early);
        JSONObject dataset=SemanticResearchDataset.build(
            OilData.fromJson(official),new ContextData(market),
            Instant.parse("2026-09-13T13:00:00Z"));
        JSONArray examples=dataset.getJSONArray("examples");
        ok(examples.length()>300,"full CPC history retained");
        ok(dataset.getInt("featureDimension")==26&&dataset.getInt("fuels")==4,
            "Android-compatible 26-feature numbers exported for all four fuels");
        boolean foundEarly=false,foundLate=false,changedTarget=false;
        for(int t=0;t<examples.length();t++){
            JSONObject row=examples.getJSONObject(t);
            LocalDate origin=LocalDate.parse(row.getString("origin"));
            LocalDate target=LocalDate.parse(row.getString("targetDate"));
            ok(target.equals(origin.plusWeeks(1)),"weekly target must be subsequent Monday");
            ok(row.getJSONArray("features").length()==4,"4 fuels per origin");
            ok(row.getJSONArray("features").getJSONArray(0).length()==26,
                "26 historical features are used at training origin");
            JSONArray articleWindow=row.getJSONArray("news");
            for(int j=0;j<articleWindow.length();j++){
                JSONObject item=articleWindow.getJSONObject(j);
                Instant cutoff=Instant.parse(row.getString("cutoff"));
                ok(!Instant.parse(item.getString("publishedAt")).isAfter(cutoff),
                    "news cannot be published in future");
                ok(!Instant.parse(item.getString("availableAt")).isAfter(cutoff),
                    "news discovered later cannot leak into early training");
                if(item.getString("id").equals(early.getString("id")))foundEarly=true;
                if(item.getString("id").equals(late.getString("id")))foundLate=true;
            }
            for(int f=0;f<4;f++){
                double delta=row.getJSONArray("nextWeekDeltas").getDouble(f);
                if(Math.abs(delta)>.049)changedTarget=true;
                ok(Double.isFinite(delta),"outcome must be finite");
            }
        }
        ok(foundEarly,"early event appears in its actual as-of training window");
        ok(!foundLate,"same old headline discovered in future is excluded");
        ok(changedTarget,"outcomes include actual adjusted CPC weeks, not just unchanged weeks");
        System.out.println("PASS: "+checks+" research dataset weekly target and as-of news assertions");
    }
}
