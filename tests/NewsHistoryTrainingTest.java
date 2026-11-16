import java.time.*;
import java.nio.file.*;
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
        // These are global logistics and conflict signals, even without any oil word.
        ok(ContextData.topics("War escalates in Eastern Europe").toString().contains("4"),
           "worldwide armed conflict affects geopolitical risk feature");
        ok(ContextData.topics("Suez Canal shipping route blocked by strikes").toString().contains("4"),
           "canal shipping blockage affects route risk feature");
        ok(ContextData.topics("Port congestion reroutes cargo vessels").toString().contains("4"),
           "congested ports and vessel rerouting affect route risk feature");
        ok(ContextData.topics("港口罷工導致航線中斷").toString().contains("4"),
           "Chinese global port disruptions are recognized");
        ok(!ContextData.topics("Panama Canal ships pass as normal").toString().contains("4"),
           "routine safe canal operation is not labelled as disruption");
        JSONObject globalConflict=ContextData.article("War escalates in Eastern Europe",
            "https://www.bbc.com/news/world-conflict-signal","BBC",
            Instant.parse("2026-10-29T00:00:00Z"),
            Instant.parse("2026-10-29T01:00:00Z"),false);
        ok(globalConflict!=null&&globalConflict.getBoolean("relevant")
            &&globalConflict.getJSONArray("topics").toString().contains("4"),
            "non-oil global war news is retained as a trainable risk event");
        JSONObject routeClosure=ContextData.article("Suez Canal shipping route blocked",
            "https://news.google.com/rss/articles/route-closure","Global maritime",
            Instant.parse("2026-10-28T00:00:00Z"),
            Instant.parse("2026-10-28T02:00:00Z"),false);
        ok(routeClosure!=null&&routeClosure.getJSONArray("topics").toString().contains("4"),
           "maritime routing problems enter the existing 26-dimensional feature");
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

        // The previous 5,000-story rolling eviction must never discard earlier news.
        JSONArray longArchive=new JSONArray();
        for(int i=0;i<5005;i++)
            longArchive.put(new JSONObject()
                .put("id","archive-"+i).put("title","Historical oil event "+i)
                .put("source","EIA").put("availableAt","2020-01-01T12:00:00Z"));
        ok(ContextData.mergeNews(new JSONArray(),longArchive).length()==5005,
           "stories older than the former 5,000-item cap remain in training");
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
        // More than 5,000 stories in one 28-day window used to abort every
        // scheduled publication. The window is now reduced by an explicit
        // priority rule, identically for cloud training and the Android payload.
        JSONArray crowded=new JSONArray();
        String[] crowdedSources={"BBC","EIA","Global maritime"};
        for(int i=0;i<5600;i++){
            String title=i%7==0?"OPEC oil production cut round "+i:i%7==1?"Crude oil market briefing "+i:"Local festival report "+i;
            crowded.put(article(title,"https://www.bbc.com/news/crowded-"+i,crowdedSources[i%3],
                Instant.parse("2026-10-10T00:00:00Z").plusSeconds(i*300L).toString(),
                Instant.parse("2026-10-10T00:00:00Z").plusSeconds(i*300L+60).toString()));
        }
        JSONObject crowdedFull=ContextData.empty().put("news",ContextData.mergeNews(new JSONArray(),crowded));
        JSONObject crowdedPublic=ContextData.publicationView(crowdedFull,Instant.parse(now));
        JSONObject selection=crowdedPublic.getJSONObject("newsSelection");
        ok(crowdedPublic.getJSONArray("news").length()==ContextData.MAX_PUBLIC_NEWS,"crowded window is reduced to the Android limit instead of failing");
        ok(selection.getBoolean("reduced")&&selection.getInt("windowCount")==5600&&selection.getInt("publishedCount")==5000,"reduction is reported, not silent");
        ok(selection.getInt("omittedEvent")==0&&selection.getInt("omittedRelevant")==0&&selection.getInt("omittedGeneral")==600,"only general news is dropped while event and oil stories all fit");
        ok(crowdedFull.getJSONArray("news").length()==5600,"training archive keeps every story");
        ContextData.Snapshot crowdedCloud=new ContextData(crowdedFull).snapshot(Instant.parse(now));
        ContextData.Snapshot crowdedApp=new ContextData(crowdedPublic).snapshot(Instant.parse(now));
        ok(crowdedCloud.count==crowdedApp.count&&crowdedCloud.articles.length()<=ContextData.MAX_PUBLIC_NEWS,"cloud and App count the same bounded stories");
        for(int i=0;i<ContextData.DIM;i++)
            ok(crowdedCloud.values[i]==crowdedApp.values[i],"crowded-window features are bit-identical on dimension "+i);
        ok(ContextData.publicationView(crowdedPublic,Instant.parse(now)).getJSONArray("news").toString().equals(crowdedPublic.getJSONArray("news").toString()),"selection is idempotent");
        ok(!ContextData.publicationView(full,Instant.parse(now)).getJSONObject("newsSelection").getBoolean("reduced"),"small windows are published unchanged");
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
        // Verify that past-news features change the actual fitted oil-price
        // model, rather than only increasing an unused JSON news counter.
        Path repo=Path.of(args.length>0?args[0]:".");
        List<OilData.Row> priceRows=OilData.weeklyThrough(
            OilData.fromJson(new JSONObject(Files.readString(repo.resolve("data/seed.json"))).getJSONArray("rows")),
            LocalDate.parse("2026-09-07"));
        Instant trainingAsOf=Instant.parse("2026-09-13T13:00:00Z");
        MultiLearner.Data newsData=new MultiLearner.Data(priceRows,ctx,trainingAsOf);
        MultiLearner.Data noNewsData=new MultiLearner.Data(priceRows,new ContextData(ContextData.empty()),trainingAsOf);
        int oldWeek=-1;
        for(int i=0;i<priceRows.size();i++)
            if(priceRows.get(i).date.equals(LocalDate.parse("2020-05-04")))oldWeek=i;
        ok(oldWeek>=12&&newsData.x[0][oldWeek][24]>0,
           "a 2020 article contributes to the 2020 weekly training feature");
        ok(noNewsData.x[0][oldWeek][24]==0,
           "the news contribution is absent without that article");
        MultiLearner.Fit fitted=MultiLearner.fit(newsData,priceRows.size()-1,0,3,3);
        MultiLearner.Fit baseline=MultiLearner.fit(noNewsData,priceRows.size()-1,0,3,3);
        double changedWeights=Math.abs(fitted.intercept-baseline.intercept);
        for(int i=0;i<fitted.weights.length;i++)
            changedWeights+=Math.abs(fitted.weights[i]-baseline.weights[i]);
        ok(changedWeights>1e-7,
           "retained old news actually affects multi-source fit coefficients");
        System.out.println("PASS: "+passed+" full-history news ingestion, causal timestamps and inference parity assertions");
    }
}
