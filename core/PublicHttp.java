package tw.oilcast.core;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPInputStream;

/** Bounded public-source reads; inherited owner also cancels RSS worker reads. */
public final class PublicHttp {
    private PublicHttp() {}
    private static final InheritableThreadLocal<Thread> OWNER=new InheritableThreadLocal<>();
    private static final ScheduledExecutorService WATCH=Executors.newScheduledThreadPool(2,r->{Thread t=new Thread(r,"source-timeout");t.setDaemon(true);return t;});
    private static final ExecutorService CLOSE=Executors.newCachedThreadPool(r->{Thread t=new Thread(r,"source-close");t.setDaemon(true);return t;});
    private static final ConcurrentMap<Thread,Set<HttpURLConnection>> ACTIVE=new ConcurrentHashMap<>();
    private static final Set<String> HOSTS=new HashSet<>(Arrays.asList("vipmbr.cpc.com.tw","fred.stlouisfed.org","www.eia.gov","feeds.feedburner.com","news.google.com","tw.news.yahoo.com","money.udn.com","feeds.bbci.co.uk","www.aljazeera.com","rss.dw.com"));
    public static void begin(){OWNER.set(Thread.currentThread());}
    public static void end(){OWNER.remove();}
    public static void cancel(Thread owner){Set<HttpURLConnection> connections=ACTIVE.get(owner);if(connections!=null)for(HttpURLConnection c:connections)CLOSE.execute(c::disconnect);}
    static URL checked(String link)throws IOException{
        URL u=new URL(link);
        if(!u.getProtocol().equals("https")||u.getUserInfo()!=null||(u.getPort()!=-1&&u.getPort()!=443)||!HOSTS.contains(u.getHost().toLowerCase(Locale.ROOT)))throw new IOException("來源轉址不在允許範圍");
        return u;
    }
    public static String read(String link)throws Exception{
        for(int attempt=0;;attempt++){
            try{return readOnce(link);}
            catch(SocketTimeoutException|EOFException|RetryableException|SocketException e){
                Thread owner=OWNER.get();if(attempt>=1||Thread.currentThread().isInterrupted()||(owner!=null&&owner.isInterrupted()))throw e;
            }
        }
    }
    private static final class RetryableException extends IOException{RetryableException(int status){super("來源 HTTP "+status);}}
    private static String readOnce(String link)throws Exception{
        Thread caller=Thread.currentThread(),owner=OWNER.get();long deadline=System.nanoTime()+35_000_000_000L;
        URL url=checked(link);
        for(int redirects=0;redirects<=3;redirects++){
            if(caller.isInterrupted()||(owner!=null&&owner.isInterrupted()))throw new InterruptedIOException("已停止來源同步");
            if(System.nanoTime()>=deadline)throw new InterruptedIOException("來源讀取逾時");
            HttpURLConnection c=(HttpURLConnection)url.openConnection();c.setConnectTimeout(15000);c.setReadTimeout(20000);c.setInstanceFollowRedirects(false);
            Thread group=owner==null?caller:owner;ACTIVE.compute(group,(t,set)->{if(set==null)set=ConcurrentHashMap.newKeySet();set.add(c);return set;});AtomicBoolean closeRequested=new AtomicBoolean();
            c.setRequestProperty("User-Agent","OilcastTaiwan/3.3.3 (energy reader)");
            c.setRequestProperty("Accept-Encoding","gzip");
            ScheduledFuture<?> watch=WATCH.scheduleAtFixedRate(()->{if((caller.isInterrupted()||(owner!=null&&owner.isInterrupted())||System.nanoTime()>=deadline)&&closeRequested.compareAndSet(false,true))CLOSE.execute(c::disconnect);},100,100,TimeUnit.MILLISECONDS);
            try{
                if(caller.isInterrupted()||(owner!=null&&owner.isInterrupted()))throw new InterruptedIOException("已停止來源同步");
                int status=c.getResponseCode();
                if(status==301||status==302||status==303||status==307||status==308){String location=c.getHeaderField("Location");if(location==null)throw new IOException("來源轉址缺少網址");url=checked(new URL(url,location).toString());continue;}
                if(status==502||status==503||status==504)throw new RetryableException(status);
                if(status!=200)throw new IOException("來源 HTTP "+status);
                ByteArrayOutputStream out=new ByteArrayOutputStream();byte[] buffer=new byte[8192];int n;
                try(InputStream raw=c.getInputStream();InputStream in="gzip".equalsIgnoreCase(c.getHeaderField("Content-Encoding"))?new GZIPInputStream(raw):raw){while((n=in.read(buffer))!=-1){if(caller.isInterrupted()||(owner!=null&&owner.isInterrupted())||System.nanoTime()>=deadline)throw new InterruptedIOException("來源讀取已逾時或停止");if(out.size()+n>OilData.MAX_BYTES)throw new IOException("來源資料過大");out.write(buffer,0,n);}}
                byte[] bytes=out.toByteArray();String head=new String(bytes,0,Math.min(bytes.length,180),StandardCharsets.US_ASCII);
                return new String(bytes,head.contains("ISO-8859-1")?StandardCharsets.ISO_8859_1:StandardCharsets.UTF_8);
            }finally{watch.cancel(false);ACTIVE.computeIfPresent(group,(t,set)->{set.remove(c);return set.isEmpty()?null:set;});c.disconnect();}
        }
        throw new IOException("來源轉址次數過多");
    }
}
