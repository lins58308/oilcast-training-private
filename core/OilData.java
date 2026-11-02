package tw.oilcast.core;

import org.json.*;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.*;
import java.time.temporal.TemporalAdjusters;
import java.util.*;
import java.util.regex.*;

/** Official effective-date observations, not predictions. Missing cells mean no update. */
public final class OilData {
    public static final String SOURCE = "https://vipmbr.cpc.com.tw/mbwebs/showhistoryprice_oil.aspx";
    public static final String HISTORICAL_SOURCE = "https://vipmbr.cpc.com.tw/mbwebs/showhistoryprice_oil2019.aspx";
    public static final ZoneId TAIPEI = ZoneId.of("Asia/Taipei");
    public static final String[] FUELS = {"92 無鉛", "95 無鉛", "98 無鉛", "超級柴油"};
    public static final int MAX_BYTES = 8 * 1024 * 1024;
    public static final int MAX_STATE_BYTES = 32 * 1024 * 1024;

    public static final class Row {
        public final LocalDate date;
        public final double[] prices;
        public Row(LocalDate d, double[] p) {
            if (p.length != 4) throw new IllegalArgumentException("油品欄位數量不正確");
            for (double v : p) if (!Double.isNaN(v) && (!Double.isFinite(v) || v < 5 || v > 100))
                throw new IllegalArgumentException("油價超出有效範圍");
            date = d; prices = p.clone();
        }
        public JSONObject json() throws JSONException {
            JSONArray p = new JSONArray();
            for (double v : prices) p.put(Double.isNaN(v) ? JSONObject.NULL : v);
            return new JSONObject().put("date", date.toString()).put("prices", p);
        }
    }

    public static LocalDate today() { return LocalDate.now(TAIPEI); }
    public static LocalDate monday(LocalDate d) { return d.with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY)); }
    public static LocalDate nextMonday(LocalDate d) { return d.with(TemporalAdjusters.next(DayOfWeek.MONDAY)); }

    public static List<Row> fromJson(JSONArray a) throws JSONException {
        if (a.length() > 5000) throw new IllegalArgumentException("資料筆數超過上限");
        List<Row> out = new ArrayList<>();
        for (int i = 0; i < a.length(); i++) {
            JSONObject o = a.getJSONObject(i); JSONArray p = o.getJSONArray("prices");
            if (p.length() != 4) throw new IllegalArgumentException("資料格式錯誤");
            double[] v = new double[4];
            for (int f = 0; f < 4; f++) v[f] = p.isNull(f) ? Double.NaN : p.getDouble(f);
            out.add(new Row(LocalDate.parse(o.getString("date")), v));
        }
        return merge(Collections.<Row>emptyList(), out);
    }

    public static JSONArray json(List<Row> rows) throws JSONException {
        JSONArray a = new JSONArray(); for (Row r : rows) a.put(r.json()); return a;
    }

    public static List<Row> parseHtml(String html) {
        if (html.length() > MAX_BYTES) throw new IllegalArgumentException("來源資料過大");
        if (!html.contains("無鉛汽油92") || !html.contains("無鉛汽油95") || !html.contains("柴油"))
            throw new IllegalArgumentException("中油頁面欄位已變更，保留原有資料");
        int tableAt = html.indexOf("id=\"MyGridView\"");
        if(tableAt<0)throw new IllegalArgumentException("中油歷史表格結構已變更");
        int tableEnd=html.indexOf("</table>",tableAt);
        if(tableEnd<0)throw new IllegalArgumentException("中油歷史表格不完整");
        html=html.substring(tableAt,tableEnd);
        Matcher headers=Pattern.compile("<th\\b[^>]*>(.*?)</th>",Pattern.CASE_INSENSITIVE|Pattern.DOTALL).matcher(html);
        String[] expected={"調價日期","無鉛汽油92","無鉛汽油95","無鉛汽油98","超級/高級柴油"};
        for(String header:expected)if(!headers.find()||!headers.group(1).replaceAll("<[^>]*>","").trim().equals(header))
            throw new IllegalArgumentException("中油欄位順序已變更，保留原有資料");
        Matcher tr = Pattern.compile("<tr\\b[^>]*>(.*?)</tr>", Pattern.CASE_INSENSITIVE | Pattern.DOTALL).matcher(html);
        Pattern td = Pattern.compile("<t[dh]\\b[^>]*>(.*?)</t[dh]>", Pattern.CASE_INSENSITIVE | Pattern.DOTALL);
        List<Row> rows = new ArrayList<>();
        while (tr.find()) {
            Matcher cells = td.matcher(tr.group(1)); List<String> values = new ArrayList<>();
            while (cells.find()) values.add(cells.group(1).replaceAll("<[^>]*>", "").replace("&nbsp;", "").replace("&#160;", "").trim());
            if (values.size() < 5 || !values.get(0).matches("[0-9]{4}/[0-9]{1,2}/[0-9]{1,2}")) continue;
            String[] parts = values.get(0).split("/");
            LocalDate d = LocalDate.of(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]), Integer.parseInt(parts[2]));
            double[] p = new double[4]; boolean any = false;
            for (int f = 0; f < 4; f++) {
                String v = values.get(f + 1);
                if (v.isEmpty() || v.equals("-") || v.equals("－")) { p[f] = Double.NaN; continue; }
                if (!v.matches("[0-9]+(?:\\.[0-9]+)?")) throw new IllegalArgumentException("來源油價欄位格式變更");
                p[f] = Double.parseDouble(v); any = true;
            }
            if (any) rows.add(new Row(d, p));
        }
        if (rows.size() < 52) throw new IllegalArgumentException("未取得足夠的官方歷史資料，保留原有資料");
        return merge(Collections.<Row>emptyList(), rows);
    }

    public static List<Row> merge(List<Row> oldRows, List<Row> newRows) {
        TreeMap<LocalDate, double[]> map = new TreeMap<>();
        for (List<Row> batch : Arrays.asList(oldRows, newRows)) for (Row r : batch) {
            double[] p = map.get(r.date);
            if (p == null) { p = new double[]{Double.NaN,Double.NaN,Double.NaN,Double.NaN}; map.put(r.date, p); }
            for (int f = 0; f < 4; f++) if (!Double.isNaN(r.prices[f])) p[f] = r.prices[f];
        }
        List<Row> out = new ArrayList<>();
        for (Map.Entry<LocalDate,double[]> e : map.entrySet()) out.add(new Row(e.getKey(), e.getValue()));
        return out;
    }

    /** Carry a posted price forward only until the last observed date; never interpolate backward. */
    public static List<Row> weekly(List<Row> rows, LocalDate cutoff) {
        List<Row> known = new ArrayList<>();
        for (Row r : rows) if (!r.date.isAfter(cutoff)) known.add(r);
        if (known.isEmpty()) return new ArrayList<>();
        LocalDate start = known.get(0).date.with(TemporalAdjusters.nextOrSame(DayOfWeek.MONDAY));
        LocalDate end = monday(known.get(known.size()-1).date);
        List<Row> out = new ArrayList<>(); int i = 0;
        double[] p = {Double.NaN,Double.NaN,Double.NaN,Double.NaN};
        for (LocalDate d = start; !d.isAfter(end); d = d.plusWeeks(1)) {
            while (i < known.size() && !known.get(i).date.isAfter(d)) {
                for (int f=0; f<4; f++) if (!Double.isNaN(known.get(i).prices[f])) p[f] = known.get(i).prices[f];
                i++;
            }
            boolean complete = true; for (double v : p) if (Double.isNaN(v)) complete = false;
            if (complete) out.add(new Row(d,p));
        }
        return out;
    }

    /**
     * Build the observed price history through the CURRENT Monday, not a future
     * effective date already listed by CPC before the week starts. After a
     * successful live CPC refresh, an unchanged week is a valid price label.
     * This is never evidence that the next Monday's forecast is already known.
     */
    public static List<Row> weeklyThrough(List<Row> rows, LocalDate cutoffMonday) {
        if (!monday(cutoffMonday).equals(cutoffMonday))
            throw new IllegalArgumentException("Weekly cutoff must be Monday");
        List<Row> out = weekly(rows, cutoffMonday);
        if (out.isEmpty()) return out;
        Row last = out.get(out.size()-1);
        for (LocalDate day = last.date.plusWeeks(1); !day.isAfter(cutoffMonday); day = day.plusWeeks(1))
            out.add(new Row(day, last.prices));
        return out;
    }

    public static Row effective(List<Row> rows, LocalDate onDate) {
        double[] p = {Double.NaN,Double.NaN,Double.NaN,Double.NaN}; LocalDate date = null;
        for (Row r : rows) {
            if (r.date.isAfter(onDate)) break;
            for (int f=0; f<4; f++) if (!Double.isNaN(r.prices[f])) p[f] = r.prices[f];
            date = r.date;
        }
        return date == null ? null : new Row(date,p);
    }

    public static String fingerprint(List<Row> rows) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(json(rows).toString().getBytes(StandardCharsets.UTF_8));
        StringBuilder s = new StringBuilder(); for (byte b : digest) s.append(String.format(Locale.US,"%02x",b & 255)); return s.toString();
    }

    public static String read(InputStream in) throws IOException { return read(in,MAX_BYTES); }
    public static String read(InputStream in,int limit) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream(); byte[] buf = new byte[8192]; int n;
        while ((n = in.read(buf)) != -1) {
            if (Thread.currentThread().isInterrupted()) throw new InterruptedIOException("作業已停止");
            if (out.size() + n > limit) throw new IOException("資料超過容量上限");
            out.write(buf,0,n);
        }
        return new String(out.toByteArray(), StandardCharsets.UTF_8);
    }


    /**
     * CPC authoritative 2019-and-earlier effective PRICE CHANGE events.
     * Pre-2006 events are irregular, sometimes partial and occasionally
     * timestamped in Chinese. Preserve original dates and missing cells.
     * Never fabricate missing fuel prices or turn international crude into CPC.
     */
    public static List<Row> parseHistoricalHtml(String html) {
        if(html.length()>MAX_BYTES)throw new IllegalArgumentException("Historical CPC page oversized");
        if(!html.contains("調價日期")||!html.contains("無鉛汽油92")||!html.contains("無鉛汽油95")
            ||!html.contains("超級/高級柴油"))throw new IllegalArgumentException("Missing official historical columns");
        Matcher tr=Pattern.compile("<tr\\b[^>]*>(.*?)</tr>",Pattern.CASE_INSENSITIVE|Pattern.DOTALL).matcher(html);
        Pattern td=Pattern.compile("<t[dh]\\b[^>]*>(.*?)</t[dh]>",Pattern.CASE_INSENSITIVE|Pattern.DOTALL);
        Pattern date=Pattern.compile("^(19|20)[0-9]{2}/[0-9]{1,2}/[0-9]{1,2}(?:\\s.*)?$");
        List<Row> result=new ArrayList<>();
        while(tr.find()) {
            Matcher cells=td.matcher(tr.group(1));List<String> values=new ArrayList<>();
            while(cells.find())values.add(cells.group(1).replaceAll("<[^>]*>","")
                 .replace("&nbsp;","").replace("&#160;","").replace("&amp;","&").trim());
            if(values.size()<5||!date.matcher(values.get(0)).matches())continue;
            String[] parts=values.get(0).split(" ")[0].split("/");
            LocalDate d=LocalDate.of(Integer.parseInt(parts[0]),Integer.parseInt(parts[1]),Integer.parseInt(parts[2]));
            if(d.isAfter(LocalDate.of(2019,12,31)))continue;
            double[] prices=new double[4];boolean any=false;
            for(int i=0;i<4;i++) {
                String s=values.get(i+1).trim();
                if(s.isEmpty()||s.equals("-")||s.equals("－")){prices[i]=Double.NaN;continue;}
                if(!s.matches("[0-9]+(?:\\.[0-9]+)?"))
                    throw new IllegalArgumentException("Unrecognized historical CPC fuel price");
                prices[i]=Double.parseDouble(s);any=true;
            }
            if(any)result.add(new Row(d,prices));
        }
        List<Row> rows=merge(Collections.<Row>emptyList(),result);
        if(rows.size()<200||rows.get(0).date.isAfter(LocalDate.of(2000,1,1))
            ||rows.get(rows.size()-1).date.isBefore(LocalDate.of(2019,12,1)))
            throw new IllegalArgumentException("Official history incomplete: 1999–2019 coverage not proven");
        return rows;
    }
    public static List<Row> fetchHistorical() throws Exception {
        return parseHistoricalHtml(PublicHttp.read(HISTORICAL_SOURCE));
    }

    public static List<Row> fetch() throws Exception {
        return parseHtml(PublicHttp.read(SOURCE));
    }
}
