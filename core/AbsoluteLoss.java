package tw.oilcast.core;

/** Training-only smooth absolute-error refinement; inference stays linear.
 * A large residual has bounded influence rather than squared-error influence.
 */
public final class AbsoluteLoss {
    private AbsoluteLoss(){}
    private static final double WIDTH=.1;
    public static double[] refine(double[][] x,double[] y,double[] baseWeights,
                                   double lambda,double intercept,double[] slopes){
        if(x.length!=y.length||x.length!=baseWeights.length||x.length<24||lambda<=0)
            throw new IllegalArgumentException("Invalid absolute-error training sample");
        int p=slopes.length,n=x.length;
        double[] beta=new double[p+1];beta[0]=intercept;
        System.arraycopy(slopes,0,beta,1,p);
        for(int i=0;i<n;i++){
            if(x[i].length!=p||!Double.isFinite(y[i])||!Double.isFinite(baseWeights[i])||baseWeights[i]<=0)
                throw new IllegalArgumentException("Invalid absolute-error observation");
            for(double value:x[i])if(!Double.isFinite(value))throw new IllegalArgumentException("Nonfinite feature");
        }
        for(int iteration=0;iteration<12;iteration++){
            if(Thread.currentThread().isInterrupted())throw new java.util.concurrent.CancellationException();
            double[][] a=new double[p+1][p+1];double[] b=new double[p+1];
            for(int i=0;i<n;i++){
                double prediction=beta[0];for(int j=0;j<p;j++)prediction+=beta[j+1]*x[i][j];
                double residual=prediction-y[i];
                double weight=baseWeights[i]*WIDTH/Math.sqrt(residual*residual+WIDTH*WIDTH);
                a[0][0]+=weight;b[0]+=weight*y[i];
                for(int j=0;j<p;j++){
                    double weighted=weight*x[i][j];
                    a[0][j+1]+=weighted;a[j+1][0]+=weighted;b[j+1]+=weighted*y[i];
                    for(int k=0;k<=j;k++)a[j+1][k+1]+=weighted*x[i][k];
                }
            }
            for(int j=1;j<=p;j++){
                a[j][j]+=lambda*WIDTH;
                for(int k=1;k<j;k++)a[k][j]=a[j][k];
            }
            double[] next=solve(a,b);double largest=0;
            for(int j=0;j<=p;j++)largest=Math.max(largest,Math.abs(next[j]-beta[j]));
            beta=next;if(largest<1e-6)break;
        }
        return beta;
    }
    private static double[] solve(double[][] a,double[] b){
        for(int c=0;c<b.length;c++){
            int pivot=c;for(int i=c+1;i<b.length;i++)if(Math.abs(a[i][c])>Math.abs(a[pivot][c]))pivot=i;
            double[] row=a[c];a[c]=a[pivot];a[pivot]=row;double t=b[c];b[c]=b[pivot];b[pivot]=t;
            double divisor=a[c][c];
            if(!Double.isFinite(divisor)||Math.abs(divisor)<1e-12)throw new IllegalArgumentException("Singular absolute-error fit");
            for(int j=c;j<b.length;j++)a[c][j]/=divisor;b[c]/=divisor;
            for(int i=0;i<b.length;i++)if(i!=c){
                double multiplier=a[i][c];for(int j=c;j<b.length;j++)a[i][j]-=multiplier*a[c][j];b[i]-=multiplier*b[c];
            }
        }
        for(double v:b)if(!Double.isFinite(v))throw new IllegalArgumentException("Nonfinite absolute-error parameter");
        return b;
    }
}
