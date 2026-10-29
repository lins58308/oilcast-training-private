import tw.oilcast.core.*;
import org.json.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.time.*;
import java.util.Arrays;

/** Read the published model without authentication using the Android downloader. */
public final class VerifyPublicDownload {
    public static void main(String[] args) throws Exception {
        Path root = Path.of(args[0]), output = Path.of(args[2]);
        String repository = args[1],tag=args.length>3?args[3]:"latest";
        if (!repository.matches("[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"))
            throw new IllegalArgumentException("Invalid repository");
        if (!tag.equals("latest")&&!tag.matches("model-[0-9]+-[0-9]+"))
            throw new IllegalArgumentException("Invalid immutable model tag");
        PublicKey key = CloudModelPackage.publicKey(Files.readString(root.resolve("model-public.txt")));
        byte[] envelopeBytes = ModelHttp.read("https://github.com/" + repository
                + "/releases/"+(tag.equals("latest")?"latest/download/":"download/"+tag+"/")+"envelope.json", 32768, null);
        JSONObject envelope = new JSONObject(new String(envelopeBytes, StandardCharsets.UTF_8));
        JSONObject manifest = CloudModelPackage.manifest(envelope, key, "oilcast-model-v1");
        byte[] payload = ModelHttp.read(CloudModelPackage.payloadUrl(manifest, repository),
                CloudModelPackage.MAX_BYTES, null);
        JSONObject data = CloudModelPackage.validate(payload, envelope, key,
                "oilcast-model-v1", Instant.now(), false);
        if (!Arrays.equals(payload, Files.readAllBytes(output.resolve("model.json"))))
            throw new IllegalStateException("Public latest model differs from this publication");
        Files.writeString(output.resolve("public-download-verification.json"), new JSONObject()
                .put("verifiedAt", Instant.now().toString()).put("signatureVerified", true)
                .put("androidDownloaderVerifiedOnJvm", true).put("androidDeviceVerified", false)
                .put("revision", data.getJSONObject("model").getInt("revision"))
                .put("payloadSha256", CloudModelPackage.sha(payload)).toString(2));
        System.out.println("Public download verified using Android downloader on JVM; revision "
                + data.getJSONObject("model").getInt("revision"));
    }
}
