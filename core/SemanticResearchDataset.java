package tw.oilcast.core;

import java.nio.file.*;
import java.time.*;
import java.util.*;
import org.json.*;

/**
 * Point-in-time research export for an offline pre-trained text encoder.
 * It never changes the signed Android prediction schema. Headlines, not
 * full copyrighted news bodies, are exported in timestamped windows.
 */
public final class SemanticResearchDataset {
    private SemanticResearchDataset(){}
    public static JSONObject build(List<OilData.Row> raw,ContextData context,Instant asOf)throws Exception{
        LocalDate today=asOf.atZone(OilData.TAIPEI).toLocalDate();
        List<OilData.Row> rows=OilData.weeklyThrough(raw,OilData.monday(today));
        if(rows.size()<90)throw new IllegalArgumentException("Need at least 90 CPC weeks");
        JSONArray examples=new JSONArray();
        Set<String> articleIds=new HashSet<>();
        int withNews=0,withMarket=0;
        for(int t=12;t<rows.size()-1;t++){
            OilData.Row r=rows.get(t),next=rows.get(t+1);
            Instant cutoff=r.date.atStartOfDay(OilData.TAIPEI).toInstant();
            if(cutoff.isAfter(asOf))throw new IllegalStateException("Training sample leaks future cutoff");
            ContextData.Snapshot snap=context.snapshot(cutoff);
            JSONArray news=new JSONArray();
            for(int i=0;i<snap.articles.length();i++){
                JSONObject original=snap.articles.getJSONObject(i);
                Instant published=Instant.parse(original.getString("publishedAt"));
                Instant available=Instant.parse(original.getString("availableAt"));
                if(published.isAfter(cutoff)||available.isAfter(cutoff))
                    throw new IllegalStateException("News published or ingested after feature cutoff");
                JSONObject item=new JSONObject().put("id",original.getString("id"))
                    .put("title",original.getString("title")).put("source",original.optString("source"))
                    .put("publishedAt",published.toString()).put("availableAt",available.toString());
                articleIds.add(item.getString("id"));news.put(item);
            }
            if(news.length()>0)withNews++;
            if(snap.values[17]>.99)withMarket++;
            JSONArray features=new JSONArray(),prices=new JSONArray(),targets=new JSONArray();
            for(int f=0;f<4;f++){
                double[] combined=snap.values.clone();
                System.arraycopy(Learner.features(rows,t,f),0,combined,0,10);
                features.put(Learner.array(combined));
                double before=r.prices[f],after=next.prices[f];
                if(!Double.isFinite(before)||!Double.isFinite(after))
                    throw new IllegalStateException("Missing official fuel price");
                prices.put(before);targets.put(after-before);
            }
            examples.put(new JSONObject().put("origin",r.date.toString())
                .put("targetDate",next.date.toString())
                .put("cutoff",cutoff.toString())
                .put("features",features).put("basePrices",prices)
                .put("nextWeekDeltas",targets).put("news",news));
        }
        return new JSONObject().put("schema",1).put("dataType","point-in-time-supervised-news-and-price")
            .put("asOf",asOf.toString()).put("through",rows.get(rows.size()-1).date.toString())
            .put("featureDimension",ContextData.DIM).put("fuels",4)
            .put("timeWindowDays",ContextData.NEWS_WINDOW_DAYS)
            .put("examples",examples).put("distinctNews",articleIds.size())
            .put("weeksWithNews",withNews).put("weeksWithMarket",withMarket)
            .put("warning","Headlines discovered later are unavailable to older training cutoffs. No full article bodies or causal labels.");
    }
    public static void main(String[] args)throws Exception{
        if(args.length!=4)throw new IllegalArgumentException("root checkpoint-or-dash output asOf");
        Path root=Path.of(args[0]),out=Path.of(args[2]);
        JSONObject state;
        if(args[1].equals("-")){
            state=new JSONObject().put("rows",new JSONObject(Files.readString(root.resolve("data/seed.json"))).getJSONArray("rows"))
                .put("multisource",new JSONObject(Files.readString(root.resolve("data/multisource.json"))));
        }else state=new JSONObject(Files.readString(Path.of(args[1])));
        if(!state.has("rows")||!state.has("multisource"))
            throw new IllegalArgumentException("Missing CPC or dated context history");
        JSONObject result=build(OilData.fromJson(state.getJSONArray("rows")),
            new ContextData(state.getJSONObject("multisource")),Instant.parse(args[3]));
        Files.createDirectories(out.toAbsolutePath().getParent());
        Files.writeString(out,result.toString());
        System.out.println(new JSONObject().put("supervisedWeeks",result.getJSONArray("examples").length())
            .put("distinctNews",result.getInt("distinctNews"))
            .put("marketWeeks",result.getInt("weeksWithMarket"))
            .put("newsWeeks",result.getInt("weeksWithNews")).toString());
    }
}
