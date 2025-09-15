import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def read_csv(p):
    df=pd.read_csv(p,header=None)
    X=df.iloc[:,0:3].to_numpy(float)
    y=df.iloc[:,3].to_numpy(float)
    return X,y

def with_intercept(X):
    return np.hstack([np.ones((X.shape[0],1)),X])

def loss(X,y,th):
    e=X@th-y
    return (e@e)/(2*len(y))

def gd(X,y,a,max_iters,tol):
    th=np.zeros(X.shape[1])
    hist=[]
    last=None
    div=False
    for i in range(max_iters):
        th-=a*(X.T@(X@th-y))/len(y)
        j=loss(X,y,th)
        hist.append(j)
        if not np.isfinite(j) or (last is not None and j>last*1.5):
            div=True
            break
        if last is not None and abs(last-j)/max(1.0,last)<tol and i>10:
            break
        last=j
    return th,hist, i+1, div

def std_fit(X):
    mu=X.mean(0); s=X.std(0); s[s==0]=1.0
    return (X-mu)/s,mu,s

def unscale(th,mu,s):
    b=th[1:]/s
    b0=th[0]-np.sum(th[1:]*mu/s)
    return np.concatenate([[b0],b])

def run_uni(X,y,alphas,max_iters,tol,std,pdf,odir):
    out={}
    for j in range(3):
        x=X[:,[j]]
        if std:
            xs,mu,s=std_fit(x); Xd=with_intercept(xs)
        else:
            mu=np.array([0.]); s=np.array([1.]); Xd=with_intercept(x)
        tries=[]
        for a in alphas:
            th,h,it,d=gd(Xd,y,a,max_iters,tol)
            tries.append((a,th,h,it,d))
        ok=[t for t in tries if not t[4]]
        best=min(ok,key=lambda t:t[2][-1]) if ok else min(tries,key=lambda t:t[2][-1])
        tho=unscale(best[1],mu,s) if std else best[1]
        xmin,xmax=x.min(),x.max()
        xs=np.linspace(xmin,xmax,200)
        ys=tho[0]+tho[1]*xs
        fig=plt.figure(); plt.scatter(x,y,s=16); plt.plot(xs,ys); plt.xlabel(f"X{j+1}"); plt.ylabel("Y"); plt.title(f"P1 fit X{j+1} (alpha={best[0]}, iters={best[3]})"); pdf.savefig(fig); plt.close(fig)
        for a,th,h,it,d in tries:
            fig=plt.figure(); plt.plot(range(1,len(h)+1),h); plt.xlabel("iter"); plt.ylabel("loss"); plt.title(f"P1 loss X{j+1} a={a} {'div' if d else ''}"); pdf.savefig(fig); plt.close(fig)
        lines=[f"P1 X{j+1} summary",f"best alpha {best[0]} iters {best[3]} final {best[2][-1]:.6g}",f"model y = {tho[0]:.6g} + ({tho[1]:.6g})*X{j+1}","alpha iters final diverged"]
        for a,th,h,it,d in tries:
            lines.append(f"{a} {it} {h[-1]:.6g} {'YES' if d else 'NO'}")
        fig=plt.figure(); plt.axis('off'); plt.text(0.02,0.98,"\n".join(lines),va='top',family='monospace'); pdf.savefig(fig); plt.close(fig)
        out[f"X{j+1}"]={"theta":tho,"best":best}
    rank=sorted([(k,v["best"][2][-1]) for k,v in out.items()], key=lambda z:z[1])
    lines=["P1 ranking (lower loss is better)"]+[f"{i+1}. {k} loss {v:.6g}" for i,(k,v) in enumerate(rank)]
    fig=plt.figure(); plt.axis('off'); plt.text(0.02,0.98,"\n".join(lines),va='top',family='monospace'); pdf.savefig(fig); plt.close(fig)
    return out

def run_multi(X,y,alphas,max_iters,tol,std,pdf):
    if std:
        Xs,mu,s=std_fit(X); Xd=with_intercept(Xs)
    else:
        mu=np.zeros(3); s=np.ones(3); Xd=with_intercept(X)
    tries=[]
    for a in alphas:
        th,h,it,d=gd(Xd,y,a,max_iters,tol)
        tries.append((a,th,h,it,d))
    ok=[t for t in tries if not t[4]]
    best=min(ok,key=lambda t:t[2][-1]) if ok else min(tries,key=lambda t:t[2][-1])
    tho=unscale(best[1],mu,s) if std else best[1]
    for a,th,h,it,d in tries:
        fig=plt.figure(); plt.plot(range(1,len(h)+1),h); plt.xlabel("iter"); plt.ylabel("loss"); plt.title(f"P2 loss a={a} {'div' if d else ''}"); pdf.savefig(fig); plt.close(fig)
    lines=["P2 summary",f"best alpha {best[0]} iters {best[3]} final {best[2][-1]:.6g}",f"model y = {tho[0]:.6g} + ({tho[1]:.6g})*X1 + ({tho[2]:.6g})*X2 + ({tho[3]:.6g})*X3","alpha iters final diverged"]
    for a,th,h,it,d in tries:
        lines.append(f"{a} {it} {h[-1]:.6g} {'YES' if d else 'NO'}")
    fig=plt.figure(); plt.axis('off'); plt.text(0.02,0.98,"\n".join(lines),va='top',family='monospace'); pdf.savefig(fig); plt.close(fig)
    return tho,best

def predict_page(pdf,th,pts):
    lines=["P2 predictions",f"y = {th[0]:.6g} + ({th[1]:.6g})*X1 + ({th[2]:.6g})*X2 + ({th[3]:.6g})*X3",""]
    for x1,x2,x3 in pts:
        y=th[0]+th[1]*x1+th[2]*x2+th[3]*x3
        lines.append(f"({x1},{x2},{x3}) -> {y:.6g}")
    fig=plt.figure(); plt.axis('off'); plt.text(0.02,0.98,"\n".join(lines),va='top',family='monospace'); pdf.savefig(fig); plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",default="D3.csv")
    ap.add_argument("--alphas",type=float,nargs="+",default=[0.1,0.05,0.02,0.01])
    ap.add_argument("--max-iters",type=int,default=5000)
    ap.add_argument("--tol",type=float,default=1e-8)
    ap.add_argument("--no-standardize",action="store_true")
    ap.add_argument("--outdir",default="results_student")
    ap.add_argument("--course",default="")
    ap.add_argument("--student",default="")
    args=ap.parse_args()

    os.makedirs(args.outdir,exist_ok=True)
    X,y=read_csv(args.csv)
    std=not args.no_standardize
    pdf_path=os.path.join(args.outdir,"hw1_results_student.pdf")
    with PdfPages(pdf_path) as pdf:
        fig=plt.figure(); plt.axis('off'); txt=f"HW1 Linear Regression with GD\n{args.course}\n{args.student}\nCSV: {args.csv}\nAlphas: {args.alphas}\nStandardize: {'ON' if std else 'OFF'}"; plt.text(0.05,0.95,txt,va='top'); pdf.savefig(fig); plt.close(fig)
        p1=run_uni(X,y,args.alphas,args.max_iters,args.tol,std,pdf,args.outdir)
        th,best=run_multi(X,y,args.alphas,args.max_iters,args.tol,std,pdf)
        predict_page(pdf,th,[(1,1,1),(2,0,4),(3,2,1)])
    print(pdf_path)

if __name__=="__main__":
    main()
