package tw.oilcast.core;
import tw.oilcast.core.*;
import org.json.*;
import java.time.*;
import java.util.*;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.security.interfaces.RSAKey;
import java.security.spec.*;

/** Signed parameter-only model contract shared by the CPU trainer and Android reader. */
public final class CloudModelPackage {
    public static final int MAX_BYTES=8*1024*1024;
    private static void require(boolean value,String message){if(!value)throw new IllegalArgumentException(message);}
    public static String sha(byte[] bytes)throws Exception{return HexFormatCompat.hex(MessageDigest.getInstance("SHA-256").digest(bytes));}
    private static byte[] pem(String text,String label){return Base64.getMimeDecoder().decode(text.replace("-----BEGIN "+label+"-----","").replace("-----END "+label+"-----",""));}
    public static PrivateKey privateKey(String text)throws Exception{return KeyFactory.getInstance("RSA").generatePrivate(new PKCS8EncodedKeySpec(pem(text,"PRIVATE KEY")));}
    public static PublicKey publicKey(String text)throws Exception{return KeyFactory.getInstance("RSA").generatePublic(new X509EncodedKeySpec(pem(text,"PUBLIC KEY")));}
    public static JSONObject envelope(JSONObject manifest,PrivateKey key,String keyId)throws Exception{
        require(key instanceof RSAKey&&((RSAKey)key).getModulus().bitLength()>=2048,"Signing key must be at least RSA2048");
        byte[] bytes=manifest.toString().getBytes(StandardCharsets.UTF_8);Signature signer=Signature.getInstance("SHA256withRSA");signer.initSign(key);signer.update(bytes);
        return new JSONObject().put("schema",1).put("keyId",keyId).put("signatureAlgorithm","SHA256withRSA").put("manifestBase64",Base64.getEncoder().encodeToString(bytes)).put("signatureBase64",Base64.getEncoder().encodeToString(signer.sign()));
    }
    public static JSONObject manifest(JSONObject envelope,PublicKey pinnedKey,String pinnedKeyId)throws Exception{
        require(envelope.getInt("schema")==1&&envelope.getString("keyId").equals(pinnedKeyId)&&envelope.getString("signatureAlgorithm").equals("SHA256withRSA"),"Untrusted key or signature algorithm");
        require(pinnedKey instanceof RSAKey&&((RSAKey)pinnedKey).getModulus().bitLength()>=2048,"Weak public key");
        require(envelope.getString("manifestBase64").length()<=22000&&envelope.getString("signatureBase64").length()<=4096,"Envelope is too large");
        byte[] signed=Base64.getDecoder().decode(envelope.getString("manifestBase64"));require(signed.length<=16384,"Manifest is too large");
        Signature verifier=Signature.getInstance("SHA256withRSA");verifier.initVerify(pinnedKey);verifier.update(signed);require(verifier.verify(Base64.getDecoder().decode(envelope.getString("signatureBase64"))),"Model signature did not verify");
        return new JSONObject(new String(signed,StandardCharsets.UTF_8));
    }
    public static JSONObject validate(byte[] payload,JSONObject envelope,PublicKey pinnedKey,String pinnedKeyId,Instant now,boolean allowDemo)throws Exception{
        require(payload.length>0&&payload.length<=MAX_BYTES,"Model package is too large");
        JSONObject manifest=manifest(envelope,pinnedKey,pinnedKeyId);
        require(manifest.getInt("schema")==1&&manifest.getInt("payloadBytes")==payload.length&&manifest.getString("payloadSha256").equals(sha(payload)),"Model checksum mismatch");
        require(!manifest.getBoolean("demo")||allowDemo,"Demo models cannot be activated in production");
        JSONObject data=new JSONObject(new String(payload,StandardCharsets.UTF_8)),model=data.getJSONObject("model");
        require(data.getInt("schema")==1&&data.getString("source").equals(OilData.SOURCE),"Unsupported model package");
        require(model.getInt("schema")==2&&model.getString("algorithm").equals(MultiLearner.VERSION)&&manifest.getString("algorithm").equals(MultiLearner.VERSION),"App algorithm upgrade is required");
        require(model.getInt("revision")>0&&model.getInt("revision")==manifest.getInt("revision"),"Invalid model revision");
        Instant trained=Instant.parse(model.getString("trainedAt")),synced=Instant.parse(data.getString("cpcSyncedAt"));
        for(Instant instant:new Instant[]{trained,synced}){long age=Duration.between(instant,now).getSeconds();require(age>=-300&&age<=36*3600,"Model or official data is stale");}
        require(model.getString("trainedAt").equals(manifest.getString("trainedAt")),"Training timestamps differ");
        LocalDate today=now.atZone(OilData.TAIPEI).toLocalDate(),through=LocalDate.parse(model.getString("trainedThrough")),target=LocalDate.parse(model.getString("targetDate"));
        require(through.toString().equals(manifest.getString("dataThrough"))&&target.toString().equals(manifest.getString("targetDate")),"Model dates differ");
        require(target.isAfter(today)&&target.equals(through.plusWeeks(1))&&!through.isBefore(OilData.monday(today))&&!through.isAfter(OilData.nextMonday(today)),"Model forecast week is stale or invalid");
        List<OilData.Row> weekly=OilData.weekly(OilData.fromJson(data.getJSONArray("rows")),OilData.nextMonday(today));
        require(weekly.size()>=80&&weekly.get(weekly.size()-1).date.equals(through),"Official rows do not match model");
        JSONArray fuels=model.getJSONArray("fuels");require(fuels.length()==4,"Four fuel models are required");
        for(int f=0;f<4;f++){
            JSONObject fuel=fuels.getJSONObject(f);require(fuel.getInt("fuel")==f,"Fuel order is invalid");
            double prediction=fuel.getDouble("predicted"),lower=fuel.getDouble("lower"),upper=fuel.getDouble("upper"),mae=fuel.getDouble("mae");
            require(Double.isFinite(prediction)&&Double.isFinite(lower)&&Double.isFinite(upper)&&Double.isFinite(mae)&&lower>=5&&upper<=100&&lower<=prediction&&prediction<=upper&&mae>=0,"Invalid forecast or error range");
            MultiLearner.Fit.from(fuel.getJSONObject("fit"));
        }
        JSONArray articles=model.getJSONObject("context").getJSONArray("articles");require(articles.length()<=5000,"Too many news references");
        for(int i=0;i<articles.length();i++)require(ContextData.allowedLink(articles.getJSONObject(i).getString("url")),"Unsafe news reference");
        require(!data.has("ledger")&&!data.has("issued")&&!data.has("authorization"),"Personal records or remote accuracy history must not be imported");
        if(data.has("cloudAccuracy"))validateAccuracy(data.getJSONObject("cloudAccuracy"));
        JSONObject outside=data.getJSONObject("multisource");validateSources(outside);
        ContextData context=new ContextData(outside);double[] external=context.snapshot(trained).values;
        for(int f=0;f<4;f++){
            double[] features=external.clone();System.arraycopy(Learner.features(weekly,weekly.size()-1,f),0,features,0,10);
            JSONObject fuel=fuels.getJSONObject(f);
            require(MultiLearner.Fit.from(fuel.getJSONObject("fit")).price(weekly.get(weekly.size()-1).prices[f],features)==fuel.getDouble("predicted"),"Model prediction does not match its parameters and public data");
        }
        return data;
    }
    private static void validateSources(JSONObject outside)throws Exception{
        JSONArray market=outside.getJSONArray("market"),news=outside.getJSONArray("news"),coverage=outside.getJSONArray("coverage");
        require(market.length()<=16000&&news.length()<=5000&&coverage.length()<=250,"Public data exceeds limit");
        for(int i=0;i<market.length();i++){
            JSONObject row=market.getJSONObject(i);LocalDate.parse(row.getString("date"));Instant.parse(row.getString("availableAt"));JSONArray values=row.getJSONArray("values");require(values.length()==3,"Invalid market row");
            for(int f=0;f<3;f++)if(!values.isNull(f)){double value=values.getDouble(f);require(Double.isFinite(value)&&(f==2?value>=5&&value<=100:value>=-100&&value<=1000),"Invalid market observation");}
        }
        for(int i=0;i<news.length();i++){
            JSONObject row=news.getJSONObject(i);require(ContextData.allowedLink(row.getString("url"))&&row.getString("title").length()<=400&&row.getString("source").length()<=200&&row.getString("id").length()<=128,"Invalid news metadata");
            Instant.parse(row.getString("publishedAt"));Instant.parse(row.getString("availableAt"));JSONArray topics=row.getJSONArray("topics");require(topics.length()<=6,"Invalid news classification");for(int j=0;j<topics.length();j++)require(topics.getInt(j)>=0&&topics.getInt(j)<6,"Invalid news classification");
        }
        for(int i=0;i<coverage.length();i++){JSONObject row=coverage.getJSONObject(i);require(!LocalDate.parse(row.getString("from")).isAfter(LocalDate.parse(row.getString("to"))),"Invalid coverage interval");}
    }
    private static void validateAccuracy(JSONObject accuracy)throws Exception{
        require(accuracy.getInt("schema")==1,"Unsupported cloud accuracy format");
        Instant.parse(accuracy.getString("updatedAt"));
        JSONArray issued=accuracy.getJSONArray("issued"),summary=accuracy.getJSONArray("perFuel");
        require(issued.length()<=260&&summary.length()==4,"Cloud accuracy history is too large");
        for(int i=0;i<issued.length();i++){
            JSONObject item=issued.getJSONObject(i);
            require("unified".equals(item.getString("mode"))&&!item.has("ledger")&&!item.has("authorization"),"Untrusted accuracy history");
            LocalDate.parse(item.getString("targetDate"));Instant.parse(item.getString("issuedAt"));
            require(item.getInt("revision")>0,"Invalid accuracy revision");
            for(String field:new String[]{"predicted","basePrices"}){
                JSONArray values=item.getJSONArray(field);require(values.length()==4,"Invalid accuracy prices");
                for(int f=0;f<4;f++)require(Double.isFinite(values.getDouble(f))&&values.getDouble(f)>=5&&values.getDouble(f)<=100,"Invalid accuracy price");
            }
            if(item.has("actual")){
                JSONArray values=item.getJSONArray("actual");require(values.length()==4,"Invalid actual prices");
                for(int f=0;f<4;f++)require(Double.isFinite(values.getDouble(f))&&values.getDouble(f)>=5&&values.getDouble(f)<=100,"Invalid actual price");
            }
        }
        for(int f=0;f<4;f++){
            JSONObject row=summary.getJSONObject(f);require(row.getInt("fuel")==f&&row.getInt("count")>=0,"Invalid accuracy summary");
            for(String field:new String[]{"mae","bias","within01","direction"})if(!row.isNull(field))require(Double.isFinite(row.getDouble(field))&&Math.abs(row.getDouble(field))<=100,"Invalid accuracy metric");
        }
    }
    public static String payloadUrl(JSONObject manifest,String repository)throws Exception{
        require(repository.matches("[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"),"Invalid repository");
        String url=manifest.getString("payloadUrl"),prefix="https://github.com/"+repository+"/releases/download/";
        require(url.startsWith(prefix)&&url.substring(prefix.length()).matches("model-[A-Za-z0-9_-]+/model\\.json"),"Untrusted model URL");return url;
    }
    private static class HexFormatCompat{static String hex(byte[] bytes){StringBuilder s=new StringBuilder();for(byte b:bytes)s.append(String.format(Locale.ROOT,"%02x",b&255));return s.toString();}}
}
