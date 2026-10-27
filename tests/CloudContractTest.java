import tw.oilcast.core.*;
import org.json.*;
import java.time.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.util.*;

public final class CloudContractTest {
    static int checks;static PrivateKey privateKey;static PublicKey publicKey;static JSONObject base,manifest;static Instant now=Instant.parse("2026-09-13T13:00:00Z");
    interface Task{void run()throws Exception;}
    static void ok(boolean b,String message){checks++;if(!b)throw new AssertionError(message);}
    static void rejects(Task task,String message)throws Exception{boolean bad=false;try{task.run();}catch(Exception e){bad=true;}ok(bad,message);}
    static JSONObject envelope(byte[] payload,JSONObject changes)throws Exception{
        JSONObject m=new JSONObject(manifest.toString()).put("payloadSha256",CloudModelPackage.sha(payload)).put("payloadBytes",payload.length);if(changes!=null)for(String key:changes.keySet())m.put(key,changes.get(key));return CloudModelPackage.envelope(m,privateKey,"oilcast-model-v1");
    }
    static JSONObject validate(JSONObject payload,JSONObject changes,Instant at,boolean demo)throws Exception{byte[] bytes=payload.toString().getBytes(StandardCharsets.UTF_8);return CloudModelPackage.validate(bytes,envelope(bytes,changes),publicKey,"oilcast-model-v1",at,demo);}
    public static void main(String[] args)throws Exception{
        Path root=Path.of(args[0]),out=root.resolve("output");base=new JSONObject(Files.readString(out.resolve("model.json")));manifest=new JSONObject(Files.readString(out.resolve("manifest.json")));
        KeyPairGenerator generator=KeyPairGenerator.getInstance("RSA");generator.initialize(2048);KeyPair pair=generator.generateKeyPair();privateKey=pair.getPrivate();publicKey=pair.getPublic();
        JSONObject result=validate(base,null,now,true);ok(result.getJSONObject("model").getString("algorithm").equals(MultiLearner.VERSION),"verified cloud package uses App algorithm");
        JSONObject cloudAccuracy=result.getJSONObject("cloudAccuracy");
        ok(cloudAccuracy.getJSONArray("perFuel").length()==4,"cloud accuracy summarizes all four fuels");
        ok(cloudAccuracy.getJSONArray("issued").length()==1,"cloud trainer records its forecast once");
        for(int fuelIndex=0;fuelIndex<4;fuelIndex++){
            JSONObject summary=cloudAccuracy.getJSONArray("perFuel").getJSONObject(fuelIndex);
            ok(summary.has("direction")&&summary.has("directionCount")&&summary.has("directionHits"),"signed cloud package contains direction hit rate and count "+fuelIndex);
        }
        JSONObject invalidAccuracy=new JSONObject(base.toString());
        invalidAccuracy.getJSONObject("cloudAccuracy").getJSONArray("issued").getJSONObject(0)
            .getJSONArray("predicted").put(0,9999);
        rejects(()->validate(invalidAccuracy,null,now,true),"signed but invalid cloud accuracy prices are rejected");
        JSONObject privateAccuracy=new JSONObject(base.toString());
        privateAccuracy.getJSONObject("cloudAccuracy").getJSONArray("issued").getJSONObject(0)
            .put("ledger",new JSONObject().put("amount",123));
        rejects(()->validate(privateAccuracy,null,now,true),"private bill data cannot be hidden in cloud accuracy history");
        String[] ids={"DCOILBRENTEU","DCOILWTICO","DEXTAUS"};
        String[] fredFiles=new String[3];
        for(int column=0;column<3;column++){
            StringBuilder csv=new StringBuilder("observation_date,"+ids[column]+"\n");
            for(int day=0;day<125;day++){
                LocalDate date=LocalDate.of(2026,1,1).plusDays(day);
                csv.append(date).append(',').append(day==10&&column==1?".":String.valueOf(30+column*10+day*.01)).append('\n');
            }
            fredFiles[column]=csv.toString();
        }
        JSONArray reunited=ContextData.parseMarket(ContextData.mergeFredSeries(fredFiles),now);
        ok(reunited.length()==125,"independent official FRED series are reunited without dropped dates");
        ok(reunited.getJSONObject(10).getJSONArray("values").isNull(1),"missing FRED observations remain missing");
        String[] broken=fredFiles.clone();broken[2]=broken[2].replace("observation_date,DEXTAUS","observation_date,UNTRUSTED");
        rejects(()->ContextData.mergeFredSeries(broken),"single-series fallback rejects unexpected FRED headers");
        rejects(()->validate(base,null,now,false),"demo cannot install in production");
        rejects(()->validate(base,null,now.plusSeconds(36*3600+1),true),"stale model rejected");
        rejects(()->validate(base,null,now.minusSeconds(301),true),"future clock skew rejected");
        byte[] bytes=base.toString().getBytes(StandardCharsets.UTF_8);JSONObject signed=envelope(bytes,null);byte[] altered=bytes.clone();altered[1]^=1;rejects(()->CloudModelPackage.validate(altered,signed,publicKey,"oilcast-model-v1",now,true),"payload tampering rejected");
        KeyPair other=generator.generateKeyPair();rejects(()->CloudModelPackage.validate(bytes,signed,other.getPublic(),"oilcast-model-v1",now,true),"untrusted signing key rejected");
        rejects(()->CloudModelPackage.validate(bytes,signed,publicKey,"unknown-id",now,true),"unknown pinned key id rejected");
        rejects(()->CloudModelPackage.validate(new byte[CloudModelPackage.MAX_BYTES+1],signed,publicKey,"oilcast-model-v1",now,true),"oversize download rejected");
        rejects(()->CloudModelPackage.validate(bytes,new JSONObject(signed.toString()).put("manifestBase64","A".repeat(22001)),publicKey,"oilcast-model-v1",now,true),"oversize manifest rejected before decoding");
        rejects(()->validate(base,new JSONObject().put("targetDate","2030-01-01"),now,true),"signed manifest and payload dates must match");
        JSONObject data=new JSONObject(base.toString());data.getJSONObject("model").put("algorithm","unknown");final JSONObject algorithm=data;rejects(()->validate(algorithm,null,now,true),"unknown algorithm rejected even with valid signature");
        for(String field:new String[]{"ledger","issued","authorization"}){data=new JSONObject(base.toString()).put(field,new JSONObject());final JSONObject privateData=data;rejects(()->validate(privateData,null,now,true),"personal data and cloud accuracy cannot overwrite phone "+field);}
        data=new JSONObject(base.toString());data.getJSONObject("model").getJSONArray("fuels").getJSONObject(1).getJSONObject("fit").put("scale",new JSONArray("[0]"));final JSONObject dimension=data;rejects(()->validate(dimension,null,now,true),"parameter dimensions rejected");
        data=new JSONObject(base.toString());data.getJSONObject("model").getJSONArray("fuels").getJSONObject(1).put("lower",99);final JSONObject band=data;rejects(()->validate(band,null,now,true),"invalid price band rejected");
        // Training parity: the Android core has not been reimplemented in a new ML framework.
        JSONObject outside=new JSONObject(Files.readString(root.resolve("data/multisource.json")));ContextData context=new ContextData(outside);
        List<OilData.Row> weekly=OilData.weekly(OilData.fromJson(base.getJSONArray("rows")),OilData.nextMonday(now.atZone(OilData.TAIPEI).toLocalDate()));
        JSONObject repeated=MultiLearner.train(weekly,context,1,"parity-fixture",now);
        for(int f=0;f<4;f++){
            JSONObject local=repeated.getJSONArray("fuels").getJSONObject(f),cloud=base.getJSONObject("model").getJSONArray("fuels").getJSONObject(f);
            for(String field:new String[]{"predicted","lower","upper","mae","family"})ok(local.getDouble(field)==cloud.getDouble(field),"CPU cloud training matches original core: "+f+" "+field);
            ok(local.getJSONObject("fit").getJSONArray("weights").toString().equals(cloud.getJSONObject("fit").getJSONArray("weights").toString()),"identical learned weights for oil "+f);
        }
        Files.writeString(out.resolve("contract-test-results.json"),new JSONObject().put("passed",true).put("assertions",checks).put("actualCloudDeployment",false).put("androidRuntimeVerified",false).put("method","Same Java model core trained on CPU with historical public-data fixtures; RSA verification, altered payloads, stale/future models, schema/dimension/range checks and model parity").toString(2));System.out.println("PASS: "+checks+" cloud model contract assertions");
    }
}
