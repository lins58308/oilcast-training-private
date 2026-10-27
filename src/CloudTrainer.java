import tw.oilcast.core.*;
import org.json.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.time.*;
import java.util.*;

/** CPU trainer using the same model source classes as Oilcast Android 3.3.5. */
public final class CloudTrainer {
    public static void main(String[] args)throws Exception{
        Path root=Path.of(args[0]),out=Path.of(args[1]);boolean live=Boolean.parseBoolean(args[2]);
        Instant asOf=Instant.parse(args[3]);Path previous=Path.of(args[4]);String keyPath=args[5];
        if(live&&!keyPath.isEmpty()&&!Files.exists(Path.of(keyPath)))throw new IllegalArgumentException("Signing key file is missing");
        JSONObject seed=new JSONObject(Files.readString(root.resolve("data/seed.json"))),outside=new JSONObject(Files.readString(root.resolve("data/multisource.json")));
        List<OilData.Row> rows=OilData.fromJson(seed.getJSONArray("rows"));JSONObject old=null;
        if(Files.exists(previous)){
            old=new JSONObject(Files.readString(previous));
            if(old.getInt("cloudStateSchema")!=1||old.has("ledger")||old.has("authorization"))throw new IllegalArgumentException("Only public cloud training checkpoints are accepted");
            rows=OilData.fromJson(old.getJSONArray("rows"));outside=old.getJSONObject("multisource");
        }
        Files.createDirectories(out);
        TrainingProgress tracker=new TrainingProgress();tracker.start();
        ProgressListener progress=new ProgressListener(){
            long last;
            public void update(String phase,int completed,int total,int fuel,int family){tracker.update(phase,completed,total,fuel,family);write();}
            public void detail(String text){tracker.detail(text);write();}
            private void write(){if(System.nanoTime()-last<100_000_000L)return;last=System.nanoTime();try{JSONObject status=tracker.json().put("updatedAt",Instant.now().toString());status.put("percent",Math.min(92,status.getInt("percent")));Path temp=out.resolve("progress.tmp");Files.writeString(temp,status.toString());Files.move(temp,out.resolve("training-status.json"),StandardCopyOption.REPLACE_EXISTING,StandardCopyOption.ATOMIC_MOVE);}catch(Exception ignored){}}
        };
        progress.update("official",0,1,-1,-1);
        Instant cpcSynced=asOf;
        if(live){
            PublicHttp.begin();try{
                List<OilData.Row> fetched=OilData.fetch();LocalDate day=OilData.today();
                fetched.removeIf(row->row.date.isAfter(OilData.nextMonday(day)));
                if(OilData.weekly(fetched,OilData.nextMonday(day)).size()<80)throw new IllegalArgumentException("CPC history is incomplete");
                List<OilData.Row> retained=new ArrayList<>();for(OilData.Row row:rows)if(row.date.isBefore(fetched.get(0).date))retained.add(row);
                rows=OilData.merge(retained,fetched);cpcSynced=Instant.now();
                System.out.println("CPC synchronized; updating public market and news sources");
                progress.update("official",1,1,-1,-1);outside=ContextData.sync(outside,Instant.now(),progress);
            }finally{PublicHttp.end();}
            asOf=Instant.now();
        }
        LocalDate day=asOf.atZone(OilData.TAIPEI).toLocalDate();List<OilData.Row> weekly=OilData.weekly(rows,OilData.nextMonday(day));
        if(weekly.size()<80||weekly.get(weekly.size()-1).date.isBefore(OilData.monday(day)))throw new IllegalArgumentException("Official data does not cover the current week; no model published");
        ContextData context=new ContextData(outside);int revision=old==null?1:old.getJSONObject("model").getInt("revision")+1;
        String hash=ContextData.sha(OilData.fingerprint(weekly)+context.fingerprint()+day);
        long started=System.nanoTime();JSONObject model=MultiLearner.train(weekly,context,revision,hash,asOf,progress);long millis=(System.nanoTime()-started)/1_000_000;
        JSONObject priorAccuracy=old==null?null:old.optJSONObject("cloudAccuracy");
        if(priorAccuracy==null&&old!=null)priorAccuracy=CloudAccuracy.update(null,old.getJSONObject("model"),rows,asOf);
        JSONObject cloudAccuracy=CloudAccuracy.update(priorAccuracy,model,rows,asOf);
        JSONObject payload=new JSONObject().put("schema",1).put("source",OilData.SOURCE).put("cpcSyncedAt",cpcSynced.toString()).put("rows",OilData.json(rows)).put("model",model).put("multisource",outside).put("sourceStates",ContextData.catalog(outside.optJSONObject("sources"))).put("cloudAccuracy",cloudAccuracy);
        byte[] bytes=payload.toString().getBytes(StandardCharsets.UTF_8);if(bytes.length>CloudModelPackage.MAX_BYTES)throw new IllegalArgumentException("Model package exceeds limit");
        JSONObject manifest=new JSONObject().put("schema",1).put("demo",!live).put("algorithm",MultiLearner.VERSION).put("revision",revision).put("trainedAt",asOf.toString()).put("dataThrough",model.getString("trainedThrough")).put("targetDate",model.getString("targetDate")).put("payloadSha256",CloudModelPackage.sha(bytes)).put("payloadBytes",bytes.length).put("payloadFile","model.json");
        if(args.length>6&&!args[6].isEmpty())manifest.put("payloadUrl",args[6]);
        Files.write(out.resolve("model.json"),bytes);Files.writeString(out.resolve("manifest.json"),manifest.toString(2));
        if(!keyPath.isEmpty())Files.writeString(out.resolve("envelope.json"),CloudModelPackage.envelope(manifest,CloudModelPackage.privateKey(Files.readString(Path.of(keyPath))),"oilcast-model-v1").toString(2));
        JSONObject checkpoint=new JSONObject().put("cloudStateSchema",1).put("rows",OilData.json(rows)).put("multisource",outside).put("model",model).put("cloudAccuracy",cloudAccuracy);
        Files.writeString(out.resolve("checkpoint.json"),checkpoint.toString());
        Files.writeString(out.resolve("training-status.json"),new JSONObject().put("outcome","running").put("phase","publish").put("percent",95).put("updatedAt",Instant.now().toString()).toString());
        JSONObject result=new JSONObject().put("algorithm",MultiLearner.VERSION).put("revision",revision).put("demo",!live).put("published",false).put("appConnected",false).put("cloudExecutionVerified",false).put("cpuTrainMillis",millis).put("modelBytes",bytes.length).put("weeklyRows",weekly.size()).put("newsRows",context.newsSize()).put("marketRows",context.marketSize()).put("targetDate",model.getString("targetDate"));
        Files.writeString(out.resolve("result.json"),result.toString(2));System.out.println(result.toString(2));
    }
}
