import java.time.*;
import java.util.*;
import org.json.*;
import tw.oilcast.core.*;

/** Regression cases for the same publication timeline used by v6 and v7. */
public final class ModelTimelineTest {
    private static int checks;
    private static void ok(boolean pass, String description) {
        checks++;
        if (!pass) throw new AssertionError(description);
    }
    private static OilData.Row row(String date, double... prices) {
        return new OilData.Row(LocalDate.parse(date), prices);
    }
    private static JSONObject model(String target, double price) {
        JSONArray fuels = new JSONArray();
        for (int i = 0; i < 4; i++)
            fuels.put(new JSONObject().put("predicted", price + i).put("family", 0));
        return new JSONObject().put("targetDate", target).put("revision", 77)
            .put("trainedAt", "2026-11-01T12:00:00Z").put("fuels", fuels);
    }
    public static void main(String[] args) throws Exception {
        List<OilData.Row> official = Arrays.asList(
            row("2026-10-19", 26, 27, 29, 23),
            row("2026-10-26", 27, 28, 30, 24),
            row("2026-11-09", 30, 31, 33, 27));
        // A next-Monday price may be posted early. It is NOT a training label.
        List<OilData.Row> sunday = OilData.weeklyThrough(official, LocalDate.parse("2026-11-02"));
        ok(sunday.get(sunday.size()-1).date.equals(LocalDate.parse("2026-11-02")),
           "training must stop at the current Monday");
        ok(sunday.get(sunday.size()-1).prices[0] == 27,
           "premature next-Monday price must not leak into training");
        // An unchanged week has a true carry-forward price, even without a CPC event.
        List<OilData.Row> unchanged = OilData.weeklyThrough(official.subList(0, 2),
            LocalDate.parse("2026-11-02"));
        ok(unchanged.get(unchanged.size()-1).prices[0] == 27,
           "no-change week must not stall training");
        ok(unchanged.size() == sunday.size(), "no-change week has the same timeline");
        boolean rejected = false;
        try { OilData.weeklyThrough(official, LocalDate.parse("2026-11-03")); }
        catch (IllegalArgumentException expected) { rejected = true; }
        ok(rejected, "non-Monday cutoff rejected");

        List<OilData.Row> sparse = official.subList(0, 2);
        Instant firstRun = Instant.parse("2026-11-01T12:00:00Z");
        JSONObject issued = CloudAccuracy.update(null, model("2026-11-09", 27), sparse, firstRun);
        JSONArray ledger = issued.getJSONArray("issued");
        ok(ledger.length() == 1 && !ledger.getJSONObject(0).has("actual"),
           "only an unresolved forecast is issued before the target");
        ok(ledger.getJSONObject(0).getString("baseDate").equals("2026-10-26"),
           "baseline must not use future CPC effective price");
        JSONObject stillPending = CloudAccuracy.update(issued, model("2026-11-16", 27), sparse,
            Instant.parse("2026-11-09T08:00:00Z"));
        ok(!stillPending.getJSONArray("issued").getJSONObject(0).has("actual"),
           "target Monday cannot be settled prematurely");
        JSONObject settled = CloudAccuracy.update(stillPending, model("2026-11-16", 27), sparse,
            Instant.parse("2026-11-10T08:00:00Z"));
        JSONObject record = settled.getJSONArray("issued").getJSONObject(0);
        ok(record.getJSONArray("actual").getDouble(0) == 27,
           "unchanged Monday must use effective CPC price");
        ok(record.getString("actualSource").equals("CPC-effective-price"),
           "settlement must record trusted observed-price provenance");
        ok(settled.getJSONArray("perFuel").getJSONObject(0).getInt("count") == 1,
           "one issued prediction contributes one observed outcome");
        JSONObject corrected = CloudAccuracy.update(settled, model("2026-11-16", 27),
            Arrays.asList(sparse.get(0), sparse.get(1), row("2026-11-09", 28, 29, 31, 25)),
            Instant.parse("2026-11-11T08:00:00Z"));
        JSONObject revised = corrected.getJSONArray("issued").getJSONObject(0);
        ok(revised.getJSONArray("firstActual").getDouble(0) == 27,
           "official correction keeps the first observed outcome");
        ok(revised.getJSONArray("actual").getDouble(0) == 28,
           "official correction updates the effective price");
        ok(corrected.getJSONArray("perFuel").getJSONObject(0).getInt("count") == 1,
           "official correction must not double-count a forecast");
        System.out.println("PASS: " + checks + " v6/v7 CPC timeline and accuracy assertions");
    }
}
