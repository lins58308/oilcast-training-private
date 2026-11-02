import java.time.*;
import java.util.*;
import org.json.*;
import tw.oilcast.core.*;

/** All retained trusted news enters the dated archive; inference only uses
 *  articles that could actually be known at the requested point in time. */
public final class NewsHistoryTrainingTest {
    private static int passed;
    private static void ok(boolean value,String why){
        passed++;
        if(!value)throw new AssertionError(why);
    }
    private static JSONObject article(String title,String url,String source,
                                      String published,String available)throws Exception {
        return ContextData.article(title,url,source,Instant.parse(published),
                                   Instant.parse(available),false);
    }
    public static void main(String[] args)throws Exception {
        String now="2026-11-03T12:00:00Z";
        JSONObject historical=article("Oil supply cuts expected",
            "https://www.eia.gov/todayinenergy/2020oil","EIA",
            "2020-05-01T12:00:00Z","2020-05-02T12:00:00Z");
        JSONObject recent=article("OPEC oil supply cuts",
            "https://www.eia.gov/todayinenergy/recent","EIA",
            "2026-10-15T12:00:00Z","2026-10-16T12:00:00Z");
        JSONObject unrelated=article("Major world sports championship",
            "https://www.bbc.com/news/sports-2026","BBC",
            "2026-10-25T12:00:00Z","2026-10-25T12:00:00Z");
        JSONObject sameTitleOtherSource=article("OPEC oil supply cuts",
            "https://www.bbc.com/news/opec-2026","BBC",
            "2026-10-17T12:00:00Z","2026-10-17T12:00:00Z");
        JSONObject future=article("Oil supply interruption tomorrow",
            "https://www.eia.gov/todayinenergy/future","EIA",
            "2026-11-04T12:00:00Z","2026-11-04T12:00:00Z");
        ok(historical!=null&&unrelated!=null&&future!=null,"all allowed feed headlines archived");
        ok(!unrelated.getBoolean("relevant")&&unrelated.getJSONArray("topics").length()==0,
            "unrelated headlines retained as general news but cannot assert oil events");
        JSONArray batch=new JSONArray();
        for(JSONObject item:new JSONObject[]{historical,recent,unrelated,sameTitleOtherSource,future})batch.put(item);
        JSONArray merged=ContextData.mergeNews(new JSONArray(),batch);
        ok(merged.length()==5,"all distinct source articles survive merge, including old news");
        ok(ContextData.mergeNews(merged,batch).length()==5,"repeated feeds do not duplicate stories");

        JSONObject full=ContextData.empty().put("news",merged);
        ContextData ctx=new ContextData(full);
        ContextData.Snapshot historicalSnap=ctx.snapshot(Instant.parse("2020-05-20T12:00:00Z"));
        ok(historicalSnap.count==1,"old news contributes to its historically available week");
        ContextData.Snapshot present=ctx.snapshot(Instant.parse(now));
        ok(present.count==3,"all source stories inside 28-day window are counted");
        ok(present.values[24]>0,"general news volume feature includes unrelated items");
        JSONObject download=ContextData.publicationView(full,Instant.parse(now));
        ok(full.getJSONArray("news").length()==5,"publication projection never mutates checkpoint");
        ok(download.getJSONArray("news").length()==3,"Android payload keeps only inference-window items");
        ContextData.Snapshot publicSnap=new ContextData(download).snapshot(Instant.parse(now));
        for(int i=0;i<ContextData.DIM;i++)
            ok(Math.abs(present.values[i]-publicSnap.values[i])<1e-10,
               "full archive and projected live inference must agree on dimension "+i);
        ContextData.Snapshot beforePublished=ctx.snapshot(Instant.parse("2026-11-02T12:00:00Z"));
        ok(beforePublished.count==3,"future news cannot leak into historical prediction");
        // A headline discovered long after it was written cannot be treated
        // as if it had been known to the historical training origin.
        JSONObject discoveredLate=article("Old oil news found today",
            "https://www.eia.gov/todayinenergy/rediscovered","EIA",
            "2020-05-01T00:00:00Z",now);
        JSONObject reread=ContextData.empty().put("news",new JSONArray().put(discoveredLate));
        ok(new ContextData(reread).snapshot(Instant.parse("2020-05-20T12:00:00Z")).count==0,
           "backfilled news cannot be used before its availability timestamp");
        String oldRss="<rss><channel><item><title>Old financial market report</title>"
           +"<link>https://www.bbc.com/news/history</link>"
           +"<pubDate>Fri, 1 Jan 2021 12:00:00 GMT</pubDate></item></channel></rss>";
        JSONArray rss=ContextData.parseRss(oldRss,"BBC",Instant.parse(now));
        ok(rss.length()==1,"trusted RSS stories older than 90 days are no longer discarded");
        ok(rss.getJSONObject(0).getString("availableAt").equals(now),
           "backfilled RSS date is not incorrectly treated as original availability");
        System.out.println("PASS: "+passed+" full-history news ingestion, causal timestamps and inference parity assertions");
    }
}
