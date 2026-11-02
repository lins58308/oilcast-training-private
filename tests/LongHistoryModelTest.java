import java.nio.file.*;
import java.time.*;
import java.util.*;
import org.json.*;
import tw.oilcast.core.*;

/** Read-only realistic 27-year fit; no signing/publishing or future CPC leakage. */
public final class LongHistoryModelTest {
  public static void main(String[] args)throws Exception{
    Path root=Path.of(".");
    JSONArray seed=new JSONObject(Files.readString(root.resolve("data/seed.json"))).getJSONArray("rows");
    List<OilData.Row> historic=OilData.fetchHistorical();
    List<OilData.Row> full=OilData.merge(historic,OilData.fromJson(seed));
    Instant asof=Instant.parse("2026-09-13T13:00:00Z");
    List<OilData.Row> weekly=OilData.weeklyThrough(full,LocalDate.of(2026,9,7));
    if(weekly.size()<1300)throw new AssertionError("Less than 25 years of weekly CPC prices");
    ContextData outside=new ContextData(new JSONObject(Files.readString(root.resolve("data/multisource.json"))));
    JSONObject model=MultiLearner.train(weekly,outside,1,"research-only",asof);
    if(model.getInt("weeks")!=weekly.size()||model.getInt("newsHistory")<=100)
      throw new AssertionError("Long-history fit lost news or weekly labels");
    JSONArray fuels=model.getJSONArray("fuels");
    for(int f=0;f<4;f++){
      JSONObject j=fuels.getJSONObject(f);
      if(j.getInt("trainingSamples")<1300)throw new AssertionError("Missing >25-year fuel training samples");
      if(j.getJSONObject("historicalDrivers").getJSONArray("associations").length()!=9)
        throw new AssertionError("News, oil and currency association evidence missing");
      if(!Double.isFinite(j.getDouble("mae")))throw new AssertionError("Invalid holdout MAE");
      System.out.println("longHistoryFuel="+f+" selectedFamily="+j.getInt("family")+
        " samples="+j.getInt("trainingSamples")+" mae="+j.getDouble("mae")+
        " candidateMae="+j.getDouble("candidateMae")+
        " baselineMae="+j.getDouble("baselineMae"));
    }
    System.out.println("PASS: 27-year CPC training, weekly="+weekly.size()+
        " news="+model.getInt("newsHistory")+" newsWeeks="+model.getInt("newsWeeks")+
        " marketWeeks="+model.getInt("marketWeeks")+
        " earliest="+weekly.get(0).date+" latest="+weekly.get(weekly.size()-1).date);
  }
}
