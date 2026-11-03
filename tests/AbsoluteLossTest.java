import tw.oilcast.core.AbsoluteLoss;

/** A few genuine extreme outcomes must not dominate typical-week MAE. */
public final class AbsoluteLossTest {
    public static void main(String[] args){
        int n=200;double[][] x=new double[n][1];double[] y=new double[n],w=new double[n];
        double squaredIntercept=0;
        for(int i=0;i<n;i++){
            x[i][0]=(i%20-9.5)/10.;y[i]=.4*x[i][0]+(i%10==0?8:0);w[i]=1;
            squaredIntercept+=y[i]/n;
        }
        double[] fit=AbsoluteLoss.refine(x,y,w,.3,squaredIntercept,new double[]{.4});
        double robustMae=0,squaredMae=0;
        for(int i=0;i<100;i++){
            double input=(i-49.5)/50.,actual=.4*input;
            robustMae+=Math.abs(fit[0]+fit[1]*input-actual)/100;
            squaredMae+=Math.abs(squaredIntercept+.4*input-actual)/100;
        }
        if(!(robustMae<.04&&robustMae<squaredMae/5))throw new AssertionError("Large residuals still dominate typical-week MAE");
        // Every row remains influential: change an early ordinary observation.
        double original=fit[0];y[1]+=.5;
        double[] changed=AbsoluteLoss.refine(x,y,w,.3,squaredIntercept,new double[]{.4});
        if(Math.abs(changed[0]-original)<1e-8)throw new AssertionError("Early history was discarded");
        y[1]=Double.NaN;
        try{AbsoluteLoss.refine(x,y,w,.3,0,new double[]{0});throw new AssertionError("Accepted nonfinite target");}
        catch(IllegalArgumentException expected){}
        System.out.println("AbsoluteLossTest passed: typical-week MAE="+robustMae+", squared-error reference="+squaredMae);
    }
}
