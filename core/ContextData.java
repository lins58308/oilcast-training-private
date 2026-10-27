package tw.oilcast.core;

import org.json.*;
import java.io.*;
import java.net.*;
import java.nio.charset.*;
import java.security.*;
import java.time.*;
import java.time.format.*;
import java.util.*;
import java.util.regex.*;
import java.util.concurrent.*;
import javax.xml.parsers.*;
import org.w3c.dom.*;
import org.xml.sax.InputSource;

/** Public market observations and attributed, dated news metadata. No article body is stored. */
public final class ContextData {
    public static final String MARKET_URL="https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU,DCOILWTICO,DEXTAUS&cosd=2019-01-01";
    public static final String EIA_RSS="https://www.eia.gov/rss/todayinenergy.xml";
    public static final String CNA_FINANCE="https://feeds.feedburner.com/rsscna/finance";
    public static final String CNA_WORLD="https://feeds.feedburner.com/rsscna/intworld";
    public static final String[] TOPICS={"供給收縮","供給增加","需求降溫","需求增強","地緣與運輸風險","庫存與政策"};
    public static final String[] NAMES={"近一週漲跌","前二週漲跌","前三週漲跌","前四週漲跌","四週平均漲跌","十二週平均漲跌","牌價波動度","相對近期均價","汽柴油交叉變動","牌價水準",
        "布蘭特短期變動","WTI 短期變動","美元／台幣短期變動","布蘭特四週變動","布蘭特日變動波動度","布蘭特／WTI 價差","原油與匯率交互變動","行情資料涵蓋率",
        "供給收縮新聞占比","供給增加新聞占比","需求降溫新聞占比","需求增強新聞占比","地緣與運輸新聞占比","庫存與政策新聞占比","新聞數量","歷史新聞涵蓋"};
    public static final int DIM=26;
    public static final String[] GROUPS={"歷史牌價","國際原油","美元／台幣","原油與匯率交互作用","新聞：供給","新聞：需求","新聞：地緣與運輸","新聞：庫存與政策","資料涵蓋與新聞量"};
    public static int group(int i){if(i<10)return 0;if(i==12)return 2;if(i==16)return 3;if(i==17||i>=24)return 8;if(i<18)return 1;if(i<20)return 4;if(i<22)return 5;if(i==22)return 6;return 7;}
    public static String unit(int i){return i>=10&&i<=16?"%":i>=18&&i<=23||i==17||i==25?"比例":i==24?"log(1+篇數)":"模型特徵";}
    private static final Pattern RELEVANT=Pattern.compile("\\b(oil|crude|petroleum|gasoline|diesel|opec|hormuz|refiner\\w*|tanker\\w*|inflation|recession|energy|fuel|brent|wti)\\b|原油|油價|石油|汽油|柴油|中油|台塑|天然氣|能源|期貨|荷姆茲|霍爾木茲|減產|增產|通膨|聯準會|美元|台幣|央行|地緣政治",Pattern.CASE_INSENSITIVE);
    private static final String[] RULES={
        "減產|供應中斷|供給中斷|停產|限產|\\b(production|output|supply)\\b.{0,45}\\b(cut\\w*|declin\\w*|decreas\\w*|fall\\w*|fell|lower|drop\\w*)\\b|\\b(cut\\w*|declin\\w*|decreas\\w*|lower|reduc\\w*)\\b.{0,40}\\b(production|output|supply)\\b",
        "增產|供給增加|供應增加|復產|\\b(production|output|supply)\\b.{0,45}\\b(increas\\w*|grow\\w*|ris\\w*|rose|record|higher)\\b|\\b(increas\\w*|grow\\w*|ris\\w*|record|higher)\\b.{0,40}\\b(production|output|supply)\\b",
        "需求疲弱|需求減|需求降|需求放緩|衰退|\\b(demand|consumption|imports)\\b.{0,40}\\b(declin\\w*|decreas\\w*|fall\\w*|fell|lower|weak\\w*|less)\\b|\\b(recession|weak demand|lower demand)\\b",
        "需求增|需求升|需求強|需求回升|\\b(demand|consumption|imports)\\b.{0,40}\\b(increas\\w*|grow\\w*|ris\\w*|rose|record|higher)\\b|\\b(strong demand|higher demand)\\b",
        "戰爭|遇襲|封鎖|制裁|荷姆茲|霍爾木茲|紅海|油輪|颶風|\\b(war|conflict|sanctions?|hormuz|attack\\w*|disrupt\\w*|tanker\\w*|hurricane|red sea|chokepoint\\w*)\\b",
        "庫存|央行|利率|通膨|聯準會|穩定機制|凍漲|吸收|補貼|\\b(inventor\\w*|stocks?|reserves?|policy|tariffs?|inflation|interest rates?|federal reserve|subsid\\w*)\\b"};
    private static final Pattern[] EVENT_RULES=Arrays.stream(RULES).map(r->Pattern.compile(r,Pattern.CASE_INSENSITIVE)).toArray(Pattern[]::new);
    private static final Pattern UNCERTAIN=Pattern.compile("\\b(not|no longer|won't|unlikely|den(?:y|ies|ied|ial)|ruled? out|reject\\w*|cancel\\w*|rumou?r\\w*|may|might|could|consider\\w*|potential)\\b|不會|不減產|不增產|未減產|未增產|取消|否認|闢謠|傳聞|可能|考慮|擬減產|擬增產|[?？]",Pattern.CASE_INSENSITIVE);
    private static final Pattern CLAUSES=Pattern.compile("[;；。！!] |[;；。！!]|\\s+\\b(?:but|while|however)\\b\\s+|但是|但|然而",Pattern.CASE_INSENSITIVE);
    public static boolean uncertain(String title){return UNCERTAIN.matcher(title.replace('’','\'')).find();}
    public static JSONArray topics(String title)throws JSONException{
        boolean[] found=new boolean[RULES.length];
        // Ambiguous clauses remain in the news list but do not assert directional events.
        // Other independent clauses can still provide evidence. Compile rules only once.
        for(String clause:CLAUSES.split(plain(title).replace('’','\''))){
            if(uncertain(clause))continue;
            for(int i=0;i<EVENT_RULES.length;i++)if(EVENT_RULES[i].matcher(clause).find())found[i]=true;
        }
        JSONArray a=new JSONArray();for(int i=0;i<found.length;i++)if(found[i])a.put(i);return a;
    }
    public static String plain(String t){
        t=t.replaceAll("(?s)<!\\[CDATA\\[(.*?)\\]\\]>","$1").replaceAll("<[^>]*>","");
        t=t.replace("&amp;","&").replace("&quot;","\"").replace("&#39;","'").replace("&apos;","'").replace("&lt;","<").replace("&gt;",">").replace("&nbsp;"," ").replace("&#8217;","’").replace("&#8211;","–");
        return t.replaceAll("\\s+"," ").trim();
    }
    public static boolean allowedLink(String link){
        if(link==null||link.length()>2048)return false;
        try{URI u=URI.create(link);String host=u.getHost();return "https".equalsIgnoreCase(u.getScheme())&&u.getUserInfo()==null&&(u.getPort()==-1||u.getPort()==443)
            &&host!=null&&Arrays.asList("www.eia.gov","www.cna.com.tw","fred.stlouisfed.org","news.google.com","tw.news.yahoo.com","money.udn.com","www.bbc.com","www.bbc.co.uk","www.aljazeera.com","www.dw.com","dw.com").contains(host.toLowerCase(Locale.US));}
        catch(Exception e){return false;}
    }
    public static JSONObject article(String title,String link,String source,Instant published,Instant available,boolean archive)throws Exception{
        title=plain(title);if(title.length()>400)title=title.substring(0,400);
        if(!allowedLink(link)||title.isEmpty()||!RELEVANT.matcher(title).find())return null;
        String id=sha(link);
        return new JSONObject().put("id",id).put("title",title).put("url",link).put("source",source)
            .put("publishedAt",published.toString()).put("availableAt",available.toString()).put("archive",archive).put("topics",topics(title)).put("uncertain",uncertain(title)).put("classifierVersion",2);
    }
    public static JSONArray parseArchive(String html,int year)throws Exception{
        if(!html.contains("Today in Energy")||!html.contains("archive.php?my="+year))throw new IOException("EIA 歷史頁格式變更");
        Matcher m=Pattern.compile("<span class=\"date\">(.*?)</span>\\s*<h2><a href=\"(detail.php\\?id=\\d+)\">(.*?)</a>",Pattern.DOTALL).matcher(html);
        JSONArray a=new JSONArray();int count=0;DateTimeFormatter df=DateTimeFormatter.ofPattern("MMMM d, uuuu",Locale.US);
        while(m.find()){
            LocalDate day=LocalDate.parse(plain(m.group(1)),df);if(day.getYear()!=year)continue;count++;
            Instant published=day.atStartOfDay(ZoneId.of("America/New_York")).toInstant();
            JSONObject item=article(m.group(3),"https://www.eia.gov/todayinenergy/"+m.group(2),"EIA",published,day.plusDays(1).atStartOfDay(ZoneId.of("America/New_York")).toInstant(),true);
            if(item!=null)a.put(item);
        }
        if(count<1)throw new IOException("EIA 歷史頁沒有有效日期");return a;
    }
    public static JSONArray parseRss(String xml,String source,Instant seen)throws Exception{
        return parseRss(xml,source,seen,new JSONObject());
    }
    public static JSONArray parseRss(String xml,String source,Instant seen,JSONObject stats)throws Exception{
        if(xml.toUpperCase(Locale.US).contains("<!DOCTYPE")||xml.toUpperCase(Locale.US).contains("<!ENTITY"))throw new IOException("不支援含外部實體的新聞來源");
        DocumentBuilderFactory f=DocumentBuilderFactory.newInstance();f.setNamespaceAware(true);f.setExpandEntityReferences(false);
        // Android's parser does not support every Xerces feature. The pre-parse
        // declaration rejection above and rejecting resolver below always apply.
        optionalFeature(f,"http://apache.org/xml/features/disallow-doctype-decl",true);
        optionalFeature(f,"http://xml.org/sax/features/external-general-entities",false);
        optionalFeature(f,"http://xml.org/sax/features/external-parameter-entities",false);
        javax.xml.parsers.DocumentBuilder builder=f.newDocumentBuilder();
        builder.setEntityResolver((publicId,systemId)->{throw new org.xml.sax.SAXException("外部實體已停用");});
        Document d=builder.parse(new InputSource(new StringReader(xml)));NodeList items=d.getElementsByTagName("item");
        if(items.getLength()==0)items=d.getElementsByTagNameNS("*","item");
        if(items.getLength()==0)items=d.getElementsByTagNameNS("*","entry");
        if(items.getLength()==0||items.getLength()>2000)throw new IOException("新聞 RSS 沒有有效項目");JSONArray out=new JSONArray();
        int rejectedLinks=0,rejectedDates=0,irrelevant=0;
        for(int i=0;i<items.getLength();i++){
            Element e=(Element)items.item(i);
            String title=tag(e,"title");
            String link=tag(e,"link").trim();
            if(link.isEmpty()){
                NodeList atomLinks=e.getElementsByTagName("atom:link");
                if(atomLinks.getLength()>0){
                    link=((Element)atomLinks.item(0)).getAttribute("href").trim();
                }
            }
            if(link.isEmpty()){
                NodeList links=e.getElementsByTagName("link");
                if(links.getLength()>0){
                    link=((Element)links.item(0)).getAttribute("href").trim();
                }
            }
            if(link.isEmpty())link=tag(e,"guid").trim();
            String date=tag(e,"pubDate").replaceAll("\\s+"," ").trim();
            if(date.isEmpty())date=tag(e,"published").replaceAll("\\s+"," ").trim();
            if(date.isEmpty())date=tag(e,"date").trim();
            if(date.isEmpty())date=tag(e,"updated").replaceAll("\\s+"," ").trim();
            Instant pub=parsePubDate(date,seen);
            if(pub==null||pub.isAfter(seen)||pub.isBefore(seen.minusSeconds(90L*86400))){rejectedDates++;continue;}
            if(!allowedLink(link)){rejectedLinks++;continue;}
            String srcName=publisher(source);
            String sourceTag=tag(e,"source");
            if(!sourceTag.isEmpty())srcName=publisher(sourceTag.trim());
            if(title.endsWith(" - "+sourceTag)&&!sourceTag.isEmpty())title=title.substring(0,title.length()-sourceTag.length()-3);
            JSONObject a=article(title,link,srcName,pub,seen,false);if(a!=null)out.put(a);else irrelevant++;
        }
        stats.put("rawCount",items.getLength()).put("rejectedLinks",rejectedLinks).put("rejectedDates",rejectedDates).put("irrelevantCount",irrelevant).put("relevantCount",out.length());
        return out;
    }
    private static Instant parsePubDate(String date,Instant fallback){
        if(date==null||date.isEmpty())return null;
        String clean=date.replace(" EST"," -0500").replace(" EDT"," -0400").replace(" CST"," +0800").replace(" GMT"," +0000").replace(" UTC"," +0000");
        DateTimeFormatter[] dfs={
            DateTimeFormatter.RFC_1123_DATE_TIME,
            DateTimeFormatter.ISO_ZONED_DATE_TIME,
            DateTimeFormatter.ISO_OFFSET_DATE_TIME,
            DateTimeFormatter.ofPattern("EEE, d MMM yyyy HH:mm:ss Z",Locale.US),
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ssXXX"),
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
        };
        for(DateTimeFormatter df:dfs){
            try{return ZonedDateTime.parse(clean,df).toInstant();}catch(Exception ignored){}
        }
        return null;
    }
    private static String tag(Element e,String name){NodeList a=e.getElementsByTagNameNS("*",name);return a.getLength()==0?"":a.item(0).getTextContent();}
    private static void optionalFeature(DocumentBuilderFactory f,String name,boolean enabled){
        try{f.setFeature(name,enabled);}catch(javax.xml.parsers.ParserConfigurationException unsupported){/* Mandatory guards are parser independent. */}
    }
    public static JSONArray parseMarket(String csv,Instant seen)throws Exception{
        String[] lines=csv.replace("\ufeff","").split("\\r?\\n");
        if(lines.length<2||!lines[0].equals("observation_date,DCOILBRENTEU,DCOILWTICO,DEXTAUS"))throw new IOException("FRED 行情欄位格式變更");
        JSONArray out=new JSONArray();
        for(int i=1;i<lines.length;i++){
            String[] c=lines[i].split(",",-1);if(c.length!=4)throw new IOException("FRED 行情欄位不完整");
            LocalDate day=LocalDate.parse(c[0]);if(day.isBefore(LocalDate.of(2019,1,1))||day.isAfter(seen.atZone(OilData.TAIPEI).toLocalDate()))continue;
            JSONArray v=new JSONArray();boolean any=false;
            for(int j=1;j<=3;j++){
                if(c[j].isEmpty()||c[j].equals(".")){v.put(JSONObject.NULL);continue;}
                double value=Double.parseDouble(c[j]);
                if(!Double.isFinite(value)||(j<3&&(value< -100||value>1000))||(j==3&&(value<5||value>100)))throw new IOException("FRED 行情數值異常");
                v.put(value);any=true;
            }
            if(any){
                Instant assumed=day.plusDays(10).atStartOfDay(ZoneOffset.UTC).toInstant();boolean estimated=assumed.isBefore(seen);
                out.put(new JSONObject().put("date",day.toString()).put("values",v).put("availableAt",(estimated?assumed:seen).toString()).put("estimated",estimated));
            }
        }
        if(out.length()<100)throw new IOException("FRED 歷史行情不足");return out;
    }
    /** FRED occasionally times out on a combined three-series graph export.
     * Reassemble three individually downloaded *official* series by date; never
     * guess missing observations or silently drop either oil/exchange series. */
    public static String mergeFredSeries(String[] csvs)throws Exception{
        String[] ids={"DCOILBRENTEU","DCOILWTICO","DEXTAUS"};
        if(csvs.length!=ids.length)throw new IOException("FRED 備援序列數量不正確");
        TreeMap<LocalDate,String[]> joined=new TreeMap<>();
        for(int i=0;i<ids.length;i++){
            String[] lines=csvs[i].replace("\ufeff","").split("\\r?\\n");
            if(lines.length<100||!lines[0].equals("observation_date,"+ids[i]))throw new IOException("FRED "+ids[i]+" 欄位格式錯誤");
            for(int j=1;j<lines.length;j++){
                String[] cols=lines[j].split(",",-1);
                if(cols.length!=2)throw new IOException("FRED "+ids[i]+" 行情欄位不完整");
                LocalDate date=LocalDate.parse(cols[0]);
                if(date.isBefore(LocalDate.of(2019,1,1)))continue;
                if(!cols[1].isEmpty()&&!cols[1].equals(".")&&!cols[1].matches("-?[0-9]+(\\.[0-9]+)?"))throw new IOException("FRED "+ids[i]+" 行情數值格式異常");
                String[] values=joined.computeIfAbsent(date,k->new String[]{".",".","."});
                values[i]=cols[1].isEmpty()?".":cols[1];
            }
        }
        if(joined.size()<100)throw new IOException("FRED 備援歷史行情不足");
        StringBuilder csv=new StringBuilder("observation_date,DCOILBRENTEU,DCOILWTICO,DEXTAUS\n");
        for(Map.Entry<LocalDate,String[]> entry:joined.entrySet()){
            String[] values=entry.getValue();
            csv.append(entry.getKey()).append(',').append(values[0]).append(',').append(values[1]).append(',').append(values[2]).append('\n');
        }
        return csv.toString();
    }
    public static JSONArray fetchMarket(Instant seen)throws Exception{
        try{return parseMarket(download(MARKET_URL),seen);}
        catch(Exception primary){
            if(Thread.currentThread().isInterrupted())throw primary;
            try{
                String[] ids={"DCOILBRENTEU","DCOILWTICO","DEXTAUS"},files=new String[3];
                for(int i=0;i<ids.length;i++)files[i]=download("https://fred.stlouisfed.org/graph/fredgraph.csv?id="+ids[i]+"&cosd=2019-01-01");
                return parseMarket(mergeFredSeries(files),seen);
            }catch(Exception backup){
                IOException error=new IOException("FRED 合併下載及三序列備援均失敗（"+primary.getClass().getSimpleName()+" / "+backup.getClass().getSimpleName()+"）",backup);
                error.addSuppressed(primary);throw error;
            }
        }
    }
    /** Preserve observed vintages: later revisions never replace values in an earlier cutoff. */
    public static JSONArray mergeMarket(JSONArray old,JSONArray incoming,Instant seen)throws Exception{
        JSONArray out=new JSONArray(old.toString());Map<String,JSONObject> latest=new HashMap<>();
        for(int i=0;i<old.length();i++){JSONObject a=old.getJSONObject(i);JSONObject p=latest.get(a.getString("date"));if(p==null||Instant.parse(a.getString("availableAt")).isAfter(Instant.parse(p.getString("availableAt"))))latest.put(a.getString("date"),a);}
        for(int i=0;i<incoming.length();i++){
            JSONObject a=incoming.getJSONObject(i),p=latest.get(a.getString("date"));
            if(p!=null&&p.getJSONArray("values").toString().equals(a.getJSONArray("values").toString()))continue;
            if(p!=null)a=new JSONObject(a.toString()).put("availableAt",seen.toString()).put("estimated",false);
            out.put(a);latest.put(a.getString("date"),a);
        }
        if(out.length()>16000)throw new IOException("行情版本超出容量上限");return out;
    }
    public static JSONArray mergeNews(JSONArray old,JSONArray incoming)throws Exception{
        LinkedHashMap<String,JSONObject> map=new LinkedHashMap<>();Set<String> titles=new HashSet<>();
        for(JSONArray batch:Arrays.asList(old,incoming))for(int i=0;i<batch.length();i++){
            JSONObject a=batch.getJSONObject(i);String normalized=normalizedTitle(a.getString("title"),a.optString("source"));
            if(map.containsKey(a.getString("id"))||titles.contains(normalized))continue;
            map.put(a.getString("id"),a);titles.add(normalized);
        }
        List<JSONObject> list=new ArrayList<>(map.values());list.sort((a,b)->Instant.parse(a.optString("availableAt")).compareTo(Instant.parse(b.optString("availableAt"))));
        JSONArray out=new JSONArray();for(int i=Math.max(0,list.size()-5000);i<list.size();i++)out.put(list.get(i));return out;
    }
    public static String sha(String s)throws Exception{byte[] b=MessageDigest.getInstance("SHA-256").digest(s.getBytes(StandardCharsets.UTF_8));StringBuilder x=new StringBuilder();for(byte v:b)x.append(String.format(Locale.US,"%02x",v&255));return x.toString();}
    public static String download(String link)throws Exception{
        return PublicHttp.read(link);
    }
    public static JSONObject empty()throws JSONException{return new JSONObject().put("market",new JSONArray()).put("news",new JSONArray()).put("coverage",new JSONArray()).put("sources",new JSONObject());}
    public static final String[][] FEEDS={
            {"eiaRss","EIA：能源焦點",EIA_RSS},
            {"cnaFinance","中央社：財經新聞",CNA_FINANCE},
            {"cnaWorld","中央社：國際焦點",CNA_WORLD},
            {"googleOilTw","全網即時：油價與中油新聞","https://news.google.com/rss/search?q=%E6%B2%B9%E5%83%B9+OR+%E5%8E%9F%E6%B2%B9+OR+%E4%B8%AD%E6%B2%B9+OR+%E6%9F%B4%E6%B2%B9&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"},
            {"googleEnergyEn","全球快訊：國際原油與OPEC","https://news.google.com/rss/search?q=crude+oil+OR+brent+crude+OR+opec+petroleum+price&hl=en-US&gl=US&ceid=US:en"},
            {"yahooFinanceTw","Yahoo奇摩：產經新聞","https://tw.news.yahoo.com/rss/finance"},
            {"udnEnergy","經濟日報：能源期貨","https://money.udn.com/rssfeed/news/1001/5591"},
            {"bbcBusiness","BBC：財經新聞","https://feeds.bbci.co.uk/news/business/rss.xml"},
            {"aljazeera","Al Jazeera：全球焦點","https://www.aljazeera.com/xml/rss/all.xml"},
            {"dwBusiness","DW：全球經濟","https://rss.dw.com/rdf/rss-en-bus"}
        };
    public static String publisher(String s){
        if(s.contains("中央社")||s.equals("CNA"))return "中央社";
        if(s.contains("EIA"))return "EIA";if(s.contains("BBC"))return "BBC";
        if(s.contains("Al Jazeera"))return "Al Jazeera";if(s.startsWith("DW"))return "DW";
        if(s.contains("經濟日報"))return "經濟日報";if(s.startsWith("Yahoo"))return "Yahoo 奇摩";
        s=plain(s);return s.length()>100?s.substring(0,100):s;
    }
    public static String normalizedTitle(String title,String source){
        String t=plain(title);if(t.endsWith(" - "+source))t=t.substring(0,t.length()-source.length()-3);
        return t.toLowerCase(Locale.US).replaceAll("[\\p{P}\\p{Z}\\s]+","");
    }
    public static JSONObject catalog(JSONObject current)throws JSONException{
        JSONObject states=current==null?new JSONObject():new JSONObject(current.toString());
        for(String[] f:FEEDS)if(!states.has(f[0]))states.put(f[0],new JSONObject().put("name",f[1]).put("pending",true));
        if(!states.has("market"))states.put("market",new JSONObject().put("name","FRED：原油與匯率").put("pending",true));
        if(!states.has("eiaArchive"))states.put("eiaArchive",new JSONObject().put("name","EIA：能源歷史報導").put("pending",true));
        return states;
    }
    public static JSONObject sync(JSONObject stored,Instant now)throws Exception{return sync(stored,now,ProgressListener.NONE);}
    public static JSONObject sync(JSONObject stored,Instant now,ProgressListener progress)throws Exception{
        JSONObject out=new JSONObject(stored.toString()),states=catalog(out.optJSONObject("sources"));
        int total=FEEDS.length+2,done=0;
        progress.update("sources",0,total,-1,-1);
        int year=now.atZone(OilData.TAIPEI).getYear();
        ExecutorService pool=Executors.newFixedThreadPool(4);CompletionService<Object[]> tasks=new ExecutorCompletionService<>(pool);
        progress.detail("正在並行同步原油、匯率與 "+FEEDS.length+" 個新聞來源；慢來源不會擋住其他來源");
        try{
            List<String[]> all=new ArrayList<>(Arrays.asList(FEEDS));all.add(new String[]{"market","FRED：原油與匯率",MARKET_URL});all.add(new String[]{"eiaArchive","EIA：能源歷史報導","https://www.eia.gov/todayinenergy/archive.php?my="+year});
            for(String[] feed:all)tasks.submit(()->{
                try{JSONObject stats=new JSONObject();String data=feed[0].equals("market")?"":download(feed[2]);JSONArray rows=feed[0].equals("market")?fetchMarket(Instant.now()):feed[0].equals("eiaArchive")?parseArchive(data,year):parseRss(data,feed[1],Instant.now(),stats);return new Object[]{feed,rows,null,stats};}
                catch(Exception e){return new Object[]{feed,null,e,null};}
            });
            for(int i=0;i<total;i++){
                Object[] result=tasks.take().get();String[] feed=(String[])result[0];Exception error=(Exception)result[2];
                if(Thread.currentThread().isInterrupted())throw new InterruptedIOException("已停止來源同步");
                if(error==null){JSONArray rows=(JSONArray)result[1];
                    if(feed[0].equals("market"))out.put("market",mergeMarket(out.getJSONArray("market"),rows,Instant.now()));
                    else{out.put("news",mergeNews(out.getJSONArray("news"),rows));if(feed[0].equals("eiaArchive")){JSONArray old=out.getJSONArray("coverage"),coverage=new JSONArray();for(int j=0;j<old.length();j++)if(old.getJSONObject(j).getInt("year")!=year)coverage.put(old.getJSONObject(j));coverage.put(new JSONObject().put("year",year).put("from",year+"-01-01").put("to",now.atZone(OilData.TAIPEI).toLocalDate().toString()));out.put("coverage",coverage);}}
                    sourceState(states,feed[0],feed[1],Instant.now(),null);JSONObject stats=(JSONObject)result[3],s=states.getJSONObject(feed[0]);if(!feed[0].equals("market")&&!feed[0].equals("eiaArchive"))for(String key:new String[]{"rawCount","rejectedLinks","rejectedDates","irrelevantCount","relevantCount"})s.put(key,stats.optInt(key));
                }
                else sourceState(states,feed[0],feed[1],Instant.now(),error);
                progress.update("sources",++done,total,-1,-1);
                progress.detail(feed[1]+(error==null?"：已處理":"：暫未更新，保留舊資料"));
            }
        }finally{pool.shutdownNow();}
        return out.put("sources",states);
    }
    private static void sourceState(JSONObject states,String id,String name,Instant now,Exception e)throws JSONException{
        JSONObject s=states.optJSONObject(id);if(s==null)s=new JSONObject();s.put("name",name).put("attemptedAt",now.toString()).put("pending",false);
        if(e==null){s.put("lastSuccess",now.toString()).put("error","");s.remove("builtin");}else s.put("error",e.getMessage()==null?"來源暫時無法讀取":e.getMessage());states.put(id,s);
    }

    private static final class Market {LocalDate date;Instant available;double[] values;}
    private static final class News {Instant published,available;JSONObject json;int[] topics;}
    private final List<Market> market=new ArrayList<>();private final List<News> news=new ArrayList<>();
    private final JSONObject raw;private final JSONArray coverage;private final Map<String,Snapshot> cache=new HashMap<>();
    public ContextData(JSONObject data)throws Exception{
        raw=data;coverage=data.getJSONArray("coverage");JSONArray m=data.getJSONArray("market"),a=data.getJSONArray("news");
        if(m.length()>16000||a.length()>5000)throw new IOException("本機資料超出容量上限");
        for(int i=0;i<m.length();i++){JSONObject j=m.getJSONObject(i);Market p=new Market();p.date=LocalDate.parse(j.getString("date"));p.available=Instant.parse(j.getString("availableAt"));p.values=new double[3];for(int k=0;k<3;k++)p.values[k]=j.getJSONArray("values").isNull(k)?Double.NaN:j.getJSONArray("values").getDouble(k);market.add(p);}
        for(int i=0;i<a.length();i++){JSONObject j=new JSONObject(a.getJSONObject(i).toString());News p=new News();p.json=j;p.published=Instant.parse(j.getString("publishedAt"));p.available=Instant.parse(j.getString("availableAt"));JSONArray old=j.getJSONArray("topics");for(int k=0;k<old.length();k++)if(old.getInt(k)<0||old.getInt(k)>=6)throw new IOException("新聞事件類型錯誤");JSONArray t=topics(j.getString("title"));j.put("topics",t).put("uncertain",uncertain(j.getString("title"))).put("classifierVersion",2);p.topics=new int[t.length()];for(int k=0;k<t.length();k++)p.topics[k]=t.getInt(k);news.add(p);}
    }
    public String fingerprint()throws Exception{return sha(raw.getJSONArray("market").toString()+raw.getJSONArray("news").toString()+coverage.toString());}
    public static final class Snapshot {
        public double[] values=new double[DIM];public JSONArray articles=new JSONArray(),quotes=new JSONArray();public String cutoff;public int count;public JSONArray publishers=new JSONArray();
        public JSONObject json()throws JSONException{return new JSONObject().put("cutoff",cutoff).put("quotes",quotes).put("articles",articles).put("newsCount",count).put("publishers",publishers).put("publisherCount",publishers.length()).put("marketCoverage",values[17]).put("newsCoverage",values[25]);}
    }
    public Snapshot snapshot(Instant asof)throws Exception{
        Snapshot cached=cache.get(asof.toString());if(cached!=null)return cached;
        Snapshot s=new Snapshot();s.cutoff=asof.toString();LocalDate day=asof.atZone(OilData.TAIPEI).toLocalDate();
        TreeMap<LocalDate,Market> selected=new TreeMap<>();
        for(Market m:market)if(!m.available.isAfter(asof)&&!m.date.isAfter(day)&&!m.date.isBefore(day.minusDays(80))){Market prev=selected.get(m.date);if(prev==null||m.available.isAfter(prev.available))selected.put(m.date,m);}
        double[][] latest=new double[3][];String[] ids={"DCOILBRENTEU","DCOILWTICO","DEXTAUS"},names={"布蘭特原油","WTI 原油","美元／台幣"};
        for(int k=0;k<3;k++){
            List<Double> seq=new ArrayList<>();LocalDate newest=null;
            for(Market m:selected.descendingMap().values())if(!Double.isNaN(m.values[k])){if(newest==null)newest=m.date;seq.add(m.values[k]);}
            latest[k]=new double[seq.size()];for(int i=0;i<seq.size();i++)latest[k][i]=seq.get(i);
            boolean valid=seq.size()>=10&&newest!=null&&!newest.isBefore(day.minusDays(18));
            if(valid){s.values[17]+=1.0/3;s.values[10+k]=pct(avg(latest[k],0,5),avg(latest[k],5,10));}
            s.quotes.put(new JSONObject().put("name",names[k]).put("series",ids[k]).put("date",newest==null?"":newest.toString()).put("value",seq.isEmpty()?JSONObject.NULL:seq.get(0)).put("unit",k==2?"台幣／美元":"美元／桶").put("change",valid?s.values[10+k]:JSONObject.NULL).put("usable",valid).put("url","https://fred.stlouisfed.org/series/"+ids[k]));
        }
        if(latest[0].length>=25){s.values[13]=pct(avg(latest[0],0,5),avg(latest[0],20,25));double mean=0;for(int i=0;i<5;i++)mean+=pct(latest[0][i],latest[0][i+1])/5;double v=0;for(int i=0;i<5;i++)v+=Math.pow(pct(latest[0][i],latest[0][i+1])-mean,2)/5;s.values[14]=Math.sqrt(v);}
        if(latest[0].length>0&&latest[1].length>0)s.values[15]=100*(latest[0][0]-latest[1][0])/Math.max(5,Math.abs(latest[0][0]));
        s.values[16]=s.values[10]*s.values[12]/100;
        List<News> visible=new ArrayList<>();Set<String> seenTitles=new HashSet<>();Map<String,double[]> bySource=new TreeMap<>();
        for(News n:news)if(!n.available.isAfter(asof)&&!n.published.isAfter(asof)&&n.published.isAfter(asof.minusSeconds(7*86400L))){
            String source=publisher(n.json.optString("source")),title=normalizedTitle(n.json.getString("title"),source);
            if(!seenTitles.add(title))continue;
            double weight=Math.exp(-Duration.between(n.published,asof).toHours()/96.0);
            double[] part=bySource.computeIfAbsent(source,k->new double[8]);part[6]+=weight;part[7]++;
            s.count++;for(int t:n.topics)part[t]+=weight;visible.add(n);
        }
        for(Map.Entry<String,double[]> entry:bySource.entrySet()){
            double[] part=entry.getValue();for(int i=0;i<6;i++)s.values[18+i]+=part[i]/part[6]/bySource.size();
            s.publishers.put(new JSONObject().put("name",entry.getKey()).put("count",(int)part[7]).put("weight",1.0/bySource.size()));
        }
        s.values[24]=Math.log1p(s.count);
        boolean complete=true;for(int i=0;i<7;i++){
            LocalDate d=day.minusDays(i);boolean covered=false;
            for(int j=0;j<coverage.length();j++){JSONObject c=coverage.getJSONObject(j);if(!d.isBefore(LocalDate.parse(c.getString("from")))&&!d.isAfter(LocalDate.parse(c.getString("to")).plusDays(2)))covered=true;}
            if(!covered)complete=false;
        }s.values[25]=complete?1:0;
        visible.sort((a,b)->b.published.compareTo(a.published));for(News n:visible)s.articles.put(n.json);
        cache.put(asof.toString(),s);return s;
    }
    private static double avg(double[] a,int from,int to){double sum=0;for(int i=from;i<to;i++)sum+=a[i];return sum/(to-from);}
    private static double pct(double a,double b){return Math.max(-100,Math.min(100,100*(a-b)/Math.max(1,Math.abs(b))));}
    public int newsSize(){return news.size();}public int marketSize(){return market.size();}
}
