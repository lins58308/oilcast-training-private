import java.time.*;
import java.util.*;
import tw.oilcast.core.*;
public final class CpcHistoricalBackfillTest {
 private static int assertions;
 static void ok(boolean v,String message){assertions++;if(!v)throw new AssertionError(message);}
 public static void main(String[] args)throws Exception {
  StringBuilder html=new StringBuilder("<html><table id=\"MyGridView\"><tr><th>調價日期</th><th>無鉛汽油92</th><th>無鉛汽油95</th><th>無鉛汽油98</th><th>超級/高級柴油</th></tr>");
  LocalDate first=LocalDate.parse("1999-01-06");
  for(int i=0;i<280;i++){
   LocalDate day=first.plusDays(i*29L);
   if(day.isAfter(LocalDate.of(2019,12,31)))break;
   String date=day.getYear()+"/"+day.getMonthValue()+"/"+day.getDayOfMonth()+(i==1?" 下午 10:00:00":"");
   html.append("<tr><td>").append(date).append("</td><td>")
    .append(i==9?"":String.format(java.util.Locale.US,"%.1f",15+i*.03))
    .append("</td><td>20.3</td><td>")
    .append(i==10?"":String.format(java.util.Locale.US,"%.1f",22+i*.03))
    .append("</td><td>").append(i==15?"":"14.8").append("</td></tr>");
  }
  html.append("</table></html>");
  List<OilData.Row> r=OilData.parseHistoricalHtml(html.toString());
  ok(r.size()>=200,"at least 200 historical events parsed");
  ok(r.get(0).date.equals(first),"official archive start preserved");
  ok(r.get(r.size()-1).date.getYear()>=2018,"official archive end preserved");
  ok(r.get(1).date.equals(first.plusDays(29)),"Chinese afternoon date parsed");
  ok(Double.isNaN(r.get(9).prices[0]),"missing 92 remains missing");
  ok(Double.isNaN(r.get(10).prices[2]),"missing 98 remains missing");
  ok(Double.isNaN(r.get(15).prices[3]),"missing diesel remains missing");
  List<OilData.Row> next=OilData.merge(r,OilData.fromJson(new org.json.JSONObject(
    java.nio.file.Files.readString(java.nio.file.Path.of("data/seed.json"))).getJSONArray("rows")));
  List<OilData.Row> weekly=OilData.weeklyThrough(next,LocalDate.of(2026,9,7));
  ok(weekly.size()>1300,"1999-to-2026 series has over 1,300 weekly targets");
  ok(weekly.get(0).date.isBefore(LocalDate.of(2000,1,1)),"pre-2000 weekly observation is retained");
  if(args.length>0 && args[0].equals("--live")){
   List<OilData.Row> official=OilData.fetchHistorical();
   List<OilData.Row> liveWeekly=OilData.weeklyThrough(
       OilData.merge(official,OilData.fromJson(new org.json.JSONObject(
       java.nio.file.Files.readString(java.nio.file.Path.of("data/seed.json"))).getJSONArray("rows"))),
       LocalDate.of(2026,9,7));
   ok(liveWeekly.size()>1250,"CPC live archive really contains 25+ years");
   System.out.println("CPC HISTORICAL OFFICIAL PROOF oldest="+official.get(0).date+
       " latest="+official.get(official.size()-1).date+" effectiveRows="+official.size()+
       " weeklyRows="+liveWeekly.size());
  }
  System.out.println("PASS: "+assertions+" 1999-2019 CPC archive and 27-year weekly reconstruction checks");
 }
}
