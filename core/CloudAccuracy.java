package tw.oilcast.core;

import java.time.Instant;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONObject;

/** Public, personal-data-free forecast-vs-CPC history retained by the cloud trainer. */
public final class CloudAccuracy {
    public static final int MAX_RECORDS = 260;
    private CloudAccuracy() {}

    private static JSONArray prices(double[] values) {
        JSONArray result = new JSONArray();
        for (double value : values) result.put(value);
        return result;
    }

    private static int direction(double value) {
        return value > 0.049 ? 1 : value < -0.049 ? -1 : 0;
    }

    public static JSONObject update(JSONObject prior, JSONObject model,
                                    List<OilData.Row> official, Instant now) throws Exception {
        JSONArray issued = prior == null ? new JSONArray() :
            new JSONArray(prior.optJSONArray("issued") == null ? "[]" : prior.getJSONArray("issued").toString());
        Map<String, OilData.Row> actual = new HashMap<>();
        for (OilData.Row row : official) actual.put(row.date.toString(), row);
        for (int i = 0; i < issued.length(); i++) {
            JSONObject item = issued.getJSONObject(i);
            OilData.Row observed = actual.get(item.getString("targetDate"));
            if (observed == null) continue;
            JSONArray revised = prices(observed.prices);
            if (!item.has("actual")) {
                item.put("actual", revised).put("verifiedAt", now.toString());
            } else if (!item.getJSONArray("actual").toString().equals(revised.toString())) {
                if (!item.has("firstActual")) item.put("firstActual", item.getJSONArray("actual"));
                item.put("actual", revised).put("correctedAt", now.toString());
            }
        }
        String target = model.getString("targetDate");
        boolean exists = false;
        for (int i = 0; i < issued.length(); i++) {
            if (issued.getJSONObject(i).getString("targetDate").equals(target)) { exists = true; break; }
        }
        LocalDate today = now.atZone(OilData.TAIPEI).toLocalDate();
        if (!exists && LocalDate.parse(target).isAfter(today) && !actual.containsKey(target)) {
            JSONArray predicted = new JSONArray(), families = new JSONArray();
            for (int i = 0; i < 4; i++) {
                JSONObject fuel = model.getJSONArray("fuels").getJSONObject(i);
                predicted.put(fuel.getDouble("predicted"));
                families.put(fuel.getInt("family"));
            }
            List<OilData.Row> weekly = OilData.weekly(official, OilData.nextMonday(today));
            if (weekly.isEmpty()) throw new IllegalArgumentException("No official baseline for accuracy");
            OilData.Row base = weekly.get(weekly.size() - 1);
            issued.put(new JSONObject().put("mode", "unified").put("issuedAt", now.toString())
                .put("targetDate", target).put("revision", model.getInt("revision"))
                .put("algorithm", MultiLearner.VERSION).put("predicted", predicted)
                .put("families", families).put("basePrices", prices(base.prices))
                .put("baseDate", base.date.toString()).put("infoCutoff", model.getString("trainedAt")));
        }
        while (issued.length() > MAX_RECORDS) issued.remove(0);
        JSONArray summary = new JSONArray();
        for (int fuel = 0; fuel < 4; fuel++) {
            int count = 0, within01 = 0, directionMatches = 0, directionCount = 0;
            double sumAbs = 0, sumBias = 0;
            for (int i = 0; i < issued.length(); i++) {
                JSONObject item = issued.getJSONObject(i);
                JSONArray recorded = item.optJSONArray("actual");
                if (recorded == null || recorded.isNull(fuel)) continue;
                double predicted = item.getJSONArray("predicted").getDouble(fuel);
                double observed = recorded.getDouble(fuel);
                double error = predicted - observed;
                if (!Double.isFinite(error)) continue;
                count++; sumAbs += Math.abs(error); sumBias += error;
                if (Math.abs(error) <= 0.1000001) within01++;
                JSONArray baseline = item.optJSONArray("basePrices");
                if (baseline != null && !baseline.isNull(fuel)) {
                    double base = baseline.getDouble(fuel);
                    if (direction(predicted - base) == direction(observed - base)) directionMatches++;
                    directionCount++;
                }
            }
            JSONObject item = new JSONObject().put("fuel", fuel).put("count", count)
                .put("mae", count == 0 ? JSONObject.NULL : sumAbs / count)
                .put("bias", count == 0 ? JSONObject.NULL : sumBias / count)
                .put("within01", count == 0 ? JSONObject.NULL : within01 * 100.0 / count)
                .put("within01Count", within01)
                .put("directionCount", directionCount)
                .put("directionHits", directionMatches)
                .put("direction", directionCount == 0 ? JSONObject.NULL : directionMatches * 100.0 / directionCount);
            summary.put(item);
        }
        return new JSONObject().put("schema", 1).put("updatedAt", now.toString())
            .put("issued", issued).put("perFuel", summary);
    }
}
