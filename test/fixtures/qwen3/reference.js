export function referenceModel(seq) {
const T=3,D=4,F=6,V=5,HQ=2,HKV=1,HD=4,NREP=HQ/HKV,QD=HQ*HD,KD=HKV*HD,EPS=1e-6,THETA=10,SCAL=1/Math.sqrt(HD);
const TOK=['on','mat','the','sat','cat'];
const tx=String.raw;
const sub=n=>String(n).split('').map(c=>'₀₁₂₃₄₅₆₇₈₉'[+c]).join('');
function rng(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
const R_=rng(20261010),rm=(r,c,s)=>[...Array(r)].map(()=>[...Array(c)].map(()=>Math.round((R_()*2-1)*s*100)/100));
const DEF={seq:[2,4,3,0],
  E:rm(V,D,1.2),G1:[1.02,0.97,1.0,1.03],Wq:rm(QD,D,0.8),Wk:rm(KD,D,0.8),Wv:rm(KD,D,0.9),Gq:[1.0,0.96,1.04,1.0],Gk:[0.98,1.02,1.0,1.01],Wo:rm(D,QD,0.5),
  G2:[1.0,0.95,1.05,1.0],
  Wg:[[0.6,-0.4,0.2,0.5],[-0.5,0.3,0.6,-0.2],[0.3,0.7,-0.4,0.4],[-0.2,-0.6,0.5,0.3],[0.7,0.1,-0.3,-0.5],[0.1,-0.3,0.8,0.6]],
  Wu:[[0.4,0.2,-0.5,0.3],[0.3,-0.6,0.2,0.5],[-0.4,0.5,0.3,0.2],[0.6,0.1,-0.2,-0.4],[-0.3,-0.4,0.6,0.1],[0.2,0.6,0.1,-0.3]],
  Wd:[[0.3,-0.2,0.4,0.1,0.2,-0.3],[-0.4,0.3,-0.1,0.2,-0.3,0.1],[0.2,0.1,0.3,-0.4,0.1,0.2],[0.4,-0.1,0.2,0.3,0.3,0.2]],
  G:[0.94,1.03,1.04,1.01],
  W:[[0.9,-0.6,0.3,0.6],[0.2,-0.8,0.6,0.2],[-0.4,0.2,0.9,0.3],[0.5,0.7,-0.6,-0.2],[-0.7,0.4,-0.3,-0.5]]};
{const OLD=['the','cat','sat','on','mat'],perm=TOK.map(w=>OLD.indexOf(w));DEF.E=perm.map(i=>DEF.E[i]);DEF.W=perm.map(i=>DEF.W[i]);}
const clone=o=>JSON.parse(JSON.stringify(o));
const trn=M=>M[0].map((_,j)=>M.map(r=>r[j]));
const PARAMS={E:0,G1:1,Wq:2,Wk:2,Wv:2,Gq:1,Gk:1,Wo:2,G2:1,Wg:2,Wu:2,Wd:2,G:1,W:2};
let P=clone(DEF);
let TS={},OPS=[],OPI={},curOp=null;
function newT(key,R,C){const n=R*C,t={key,R,C,n,v:new Float64Array(n),g:new Float64Array(n),dep:null,op:null};TS[key]=t;return t;}
function leaf(key,M){const A=Array.isArray(M[0])?M:[M],t=newT(key,A.length,A[0].length);A.forEach((r,i)=>r.forEach((v,j)=>t.v[i*t.C+j]=v));t.leaf=true;return t;}
function op(id,mod){curOp={id,mod,outs:[],ins:[]};OPS.push(curOp);OPI[id]=curOp;}
function def(key,R,C,fn){
  const t=newT(key,R,C);t.dep=new Array(t.n);t.op=curOp.id;curOp.outs.push(key);
  for(let r=0;r<R;r++)for(let c=0;c<C;c++){const i=r*C+c,[v,dep]=fn(r,c);t.v[i]=v;t.dep[i]=dep;for(const d of dep)if(!curOp.ins.includes(d[0]))curOp.ins.push(d[0]);}
  return t;
}
const VV=(k,i)=>TS[k].v[i];
function mm(key,A,B,R,K,N,ai,bi){return def(key,R,N,(t,n)=>{let s=0;const dep=[];for(let k=0;k<K;k++){const a=VV(A,ai(t,k)),b=VV(B,bi(k,n));s+=a*b;dep.push([A,ai(t,k),b],[B,bi(k,n),a]);}return[s,dep];});}
function rms(ids,mod,specs){
  op(ids[0],mod);for(const s of specs){const X=TS[s.X];def(s.sq,X.R,X.C,(r,c)=>{const i=r*X.C+c,x=X.v[i];return[x*x,[[s.X,i,2*x]]];});}
  op(ids[1],mod);for(const s of specs){const X=TS[s.X],ng=X.C/s.gs;
    def(s.ms,X.R,ng,(r,h)=>{let a=0;const dep=[];for(let j=0;j<s.gs;j++){const i=r*X.C+h*s.gs+j;a+=VV(s.sq,i);dep.push([s.sq,i,1/s.gs]);}return[a/s.gs,dep];});
    def(s.r,X.R,ng,(r,h)=>{const i=r*ng+h,rv=1/Math.sqrt(VV(s.ms,i)+EPS);return[rv,[[s.ms,i,-rv*rv*rv/2]]];});}
  op(ids[2],mod);for(const s of specs){const X=TS[s.X],ng=X.C/s.gs;def(s.ah,X.R,X.C,(r,c)=>{const i=r*X.C+c,ri=r*ng+Math.floor(c/s.gs),x=X.v[i],rv=VV(s.r,ri);return[x*rv,[[s.X,i,rv],[s.r,ri,x]]];});}
  op(ids[3],mod);for(const s of specs){const X=TS[s.X];def(s.out,X.R,X.C,(r,c)=>{const i=r*X.C+c,j=c%s.gs,a=VV(s.ah,i),g=VV(s.G,j);return[a*g,[[s.ah,i,g],[s.G,j,a]]];});}
}
const range=n=>[...Array(n).keys()];
function forward(){
  TS={};OPS=[];OPI={};
  const ids=P.seq.slice(0,T),lab=P.seq.slice(1,T+1);
  leaf('E',P.E);leaf('G1',P.G1);leaf('Wq',trn(P.Wq));leaf('Wk',trn(P.Wk));leaf('Wv',trn(P.Wv));leaf('Gq',P.Gq);leaf('Gk',P.Gk);leaf('Wo',trn(P.Wo));
  leaf('G2',P.G2);leaf('Wg',trn(P.Wg));leaf('Wu',trn(P.Wu));leaf('Wd',trn(P.Wd));leaf('G',P.G);leaf('W',trn(P.W));
  const inv=range(HD/2).map(i=>1/Math.pow(THETA,2*i/HD));
  leaf('cos',range(T).map(t=>range(HD).map(c=>Math.cos(t*inv[c%(HD/2)]))));
  leaf('sin',range(T).map(t=>range(HD).map(c=>Math.sin(t*inv[c%(HD/2)]))));
  leaf('M',range(T).map(t=>range(T).map(s=>s>t?-Infinity:0)));
  op('emb',null);def('e',T,D,(t,k)=>{const i=ids[t]*D+k;return[VV('E',i),[['E',i,1]]];});
  op('pre','pre');def('x0',T,D,(t,k)=>{const i=t*D+k;return[VV('e',i),[['e',i,1]]];});
  rms(['r1sq','r1rs','r1ah','r1n'],'rms1',[{X:'x0',G:'G1',sq:'sq1',ms:'ms1',r:'r1',ah:'ah1',out:'n1',gs:D}]);
  op('qkv','attn');mm('q','n1','Wq',T,D,QD,(t,k)=>t*D+k,(k,n)=>k*QD+n);mm('k','n1','Wk',T,D,KD,(t,k)=>t*D+k,(k,n)=>k*KD+n);mm('v','n1','Wv',T,D,KD,(t,k)=>t*D+k,(k,n)=>k*KD+n);
  rms(['qksq','qkrs','qkah','qkn'],'qkn',[{X:'q',G:'Gq',sq:'qsq',ms:'qms',r:'qrr',ah:'qah',out:'qn',gs:HD},{X:'k',G:'Gk',sq:'ksq',ms:'kms',r:'krr',ah:'kah',out:'kn',gs:HD}]);
  op('qkc','rope');for(const [X,O] of [['qn','qc'],['kn','kc']]){const S=TS[X];def(O,T,S.C,(t,c)=>{const i=t*S.C+c,j=t*HD+c%HD,x=S.v[i],w=VV('cos',j);return[x*w,[[X,i,w],['cos',j,x]]];});}
  op('rh','rope');for(const [X,O] of [['qn','qrh'],['kn','krh']]){const S=TS[X],h2=HD/2;def(O,T,S.C,(t,c)=>{const j=c%HD,b=t*S.C+c-j;if(j<h2){const i=b+j+h2;return[-S.v[i],[[X,i,-1]]];}const i=b+j-h2;return[S.v[i],[[X,i,1]]];});}
  op('qks','rope');for(const [X,O] of [['qrh','qs'],['krh','ks']]){const S=TS[X];def(O,T,S.C,(t,c)=>{const i=t*S.C+c,j=t*HD+c%HD,x=S.v[i],w=VV('sin',j);return[x*w,[[X,i,w],['sin',j,x]]];});}
  op('qkr','rope');for(const [A,B,O] of [['qc','qs','qr'],['kc','ks','kr']]){const S=TS[A];def(O,T,S.C,(t,c)=>{const i=t*S.C+c;return[VV(A,i)+VV(B,i),[[A,i,1],[B,i,1]]];});}
  op('krep','attn');for(let h=0;h<HQ;h++){const kv=Math.floor(h/NREP);def('kp'+h,T,HD,(t,c)=>{const i=t*KD+kv*HD+c;return[VV('kr',i),[['kr',i,1]]];});}
  op('vrep','attn');for(let h=0;h<HQ;h++){const kv=Math.floor(h/NREP);def('V'+h,T,HD,(t,c)=>{const i=t*KD+kv*HD+c;return[VV('v',i),[['v',i,1]]];});}
  op('kT','attn');for(let h=0;h<HQ;h++)def('kT'+h,HD,T,(c,s)=>{const i=s*HD+c;return[VV('kp'+h,i),[['kp'+h,i,1]]];});
  op('S','attn');for(let h=0;h<HQ;h++)mm('S'+h,'qr','kT'+h,T,HD,T,(t,c)=>t*QD+h*HD+c,(c,s)=>c*T+s);
  op('Ssc','attn');for(let h=0;h<HQ;h++)def('Ss'+h,T,T,(t,s)=>{const i=t*T+s;return[VV('S'+h,i)*SCAL,[['S'+h,i,SCAL]]];});
  op('mask','attn');for(let h=0;h<HQ;h++)def('Sm'+h,T,T,(t,s)=>{const i=t*T+s;return[VV('Ss'+h,i)+VV('M',i),[['Ss'+h,i,1],['M',i,1]]];});
  op('exp','attn');for(let h=0;h<HQ;h++)def('Se'+h,T,T,(t,s)=>{const i=t*T+s,e=Math.exp(VV('Sm'+h,i));return[e,[['Sm'+h,i,e]]];});
  op('sum','attn');for(let h=0;h<HQ;h++)def('Z'+h,T,1,t=>{let a=0;const dep=[];for(let s=0;s<T;s++){a+=VV('Se'+h,t*T+s);dep.push(['Se'+h,t*T+s,1]);}return[a,dep];});
  op('div','attn');for(let h=0;h<HQ;h++)def('P'+h,T,T,(t,s)=>{const i=t*T+s,e=VV('Se'+h,i),z=VV('Z'+h,t);return[e/z,[['Se'+h,i,1/z],['Z'+h,t,-e/(z*z)]]];});
  op('PV','attn');for(let h=0;h<HQ;h++)mm('O'+h,'P'+h,'V'+h,T,T,HD,(t,s)=>t*T+s,(s,c)=>s*HD+c);
  op('cat','attn');def('O',T,QD,(t,c)=>{const h=Math.floor(c/HD),i=t*HD+c%HD;return[VV('O'+h,i),[['O'+h,i,1]]];});
  op('oproj','attn');mm('ao','O','Wo',T,QD,D,(t,j)=>t*QD+j,(j,c)=>j*D+c);
  op('res1','dec');def('a',T,D,(t,k)=>{const i=t*D+k;return[VV('x0',i)+VV('ao',i),[['x0',i,1],['ao',i,1]]];});
  rms(['r2sq','r2rs','r2ah','r2n'],'rms2',[{X:'a',G:'G2',sq:'sq2',ms:'ms2',r:'r2',ah:'ah2',out:'n',gs:D}]);
  op('gu','mlp');mm('gate','n','Wg',T,D,F,(t,k)=>t*D+k,(k,j)=>k*F+j);mm('up','n','Wu',T,D,F,(t,k)=>t*D+k,(k,j)=>k*F+j);
  op('silu','mlp');def('s',T,F,(t,j)=>{const i=t*F+j,v=VV('gate',i),sg=1/(1+Math.exp(-v));return[v*sg,[['gate',i,sg*(1+v*(1-sg))]]];});
  op('mul','mlp');def('m',T,F,(t,j)=>{const i=t*F+j,s=VV('s',i),u=VV('up',i);return[s*u,[['s',i,u],['up',i,s]]];});
  op('dn','mlp');mm('down','m','Wd',T,F,D,(t,j)=>t*F+j,(j,c)=>j*D+c);
  op('res2','dec');def('x',T,D,(t,k)=>{const i=t*D+k;return[VV('a',i)+VV('down',i),[['a',i,1],['down',i,1]]];});
  rms(['fsq','frs','fxh','fh'],'rms',[{X:'x',G:'G',sq:'sq',ms:'ms',r:'r',ah:'xh',out:'h',gs:D}]);
  op('lm',null);mm('z','h','W',T,D,V,(t,k)=>t*D+k,(k,i)=>k*V+i);
  op('lse',null);def('L',T,1,t=>{let mx=-Infinity;for(let i=0;i<V;i++)mx=Math.max(mx,VV('z',t*V+i));let se=0;for(let i=0;i<V;i++)se+=Math.exp(VV('z',t*V+i)-mx);const L=mx+Math.log(se);
    return[L,range(V).map(i=>['z',t*V+i,Math.exp(VV('z',t*V+i)-L)])];});
  op('lp',null);def('lp',T,V,(t,i)=>[VV('z',t*V+i)-VV('L',t),[['z',t*V+i,1],['L',t,-1]]]);
  op('nll',null);def('lt',T,1,t=>{const i=t*V+lab[t];return[-VV('lp',i),[['lp',i,-1]]];});
  op('mean',null);def('loss',1,1,()=>{let a=0;const dep=[];for(let t=0;t<T;t++){a+=VV('lt',t);dep.push(['lt',t,1/T]);}return[a/T,dep];});
}
function backward(){
  for(const k in TS)TS[k].g.fill(0);TS.loss.g[0]=1;
  for(let o=OPS.length-1;o>=0;o--){const op=OPS[o];
    for(let q=op.outs.length-1;q>=0;q--){const t=TS[op.outs[q]];for(let i=0;i<t.n;i++){const g=t.g[i];if(g===0)continue;for(const [k,j,d] of t.dep[i])if(d!==0)TS[k].g[j]+=g*d;}}}
}
const MODS={
  pre:{parent:null,t:'⋯ 第 1 到 N−1 层',s:'Qwen3DecoderLayer × (N−1)',io:[['e','x0']],fixed:true},
  dec:{parent:null,t:'Qwen3DecoderLayer',s:'第 N 层',io:[['x0','x']]},
  rms1:{parent:'dec',t:'input_layernorm',s:'Qwen3RMSNorm',io:[['x0','n1']]},
  attn:{parent:'dec',t:'self_attn',s:'Qwen3Attention',io:[['n1','aoa']]},
  qkn:{parent:'attn',t:'q_norm · k_norm',s:'Qwen3RMSNorm，按 head_dim',io:[['q','qn'],['k','kn']]},
  rope:{parent:'attn',t:'apply_rotary_pos_emb',s:'RoPE',io:[['qn','qr'],['kn','kr']]},
  rms2:{parent:'dec',t:'post_attention_layernorm',s:'Qwen3RMSNorm',io:[['a','n']]},
  mlp:{parent:'dec',t:'mlp',s:'Qwen3MLP',io:[['n','downa']]},
  rms:{parent:null,t:'model.norm',s:'Qwen3RMSNorm',io:[['x','h']]}};
const ALIAS={n1a:'n1',na:'n',ha:'h',aoa:'ao',downa:'down'};
const fold={pre:true,dec:false,rms1:false,attn:false,qkn:false,rope:false,rms2:false,mlp:false,rms:false};
const inMod=(m,target)=>{for(let x=m;x;x=MODS[x].parent)if(x===target)return true;return false;};
const hiddenBy=m=>{for(let x=m;x;x=MODS[x].parent)if(fold[x])return true;return false;};
const outerFold=m=>{let r=null;for(let x=m;x;x=MODS[x].parent)if(fold[x])r=x;return r;};
const showBox=m=>fold[m]&&!hiddenBy(MODS[m].parent);
function jac(mod,inK,outK){
  if(typeof ALIAS!=='undefined'){inK=ALIAS[inK]||inK;outK=ALIAS[outK]||outK;}
  const ops=OPS.filter(o=>inMod(o.mod,mod)),O=TS[outK],I=TS[inK],J=[];
  for(let i=0;i<O.n;i++){const gl={[outK]:new Float64Array(O.n)};gl[outK][i]=1;
    for(let a=ops.length-1;a>=0;a--){const op=ops[a];for(let q=op.outs.length-1;q>=0;q--){const k=op.outs[q],gg=gl[k];if(!gg)continue;const t=TS[k];
      for(let e=0;e<t.n;e++){const g=gg[e];if(!g)continue;for(const [kk,j,d] of t.dep[e])if(d!==0)(gl[kk]||(gl[kk]=new Float64Array(TS[kk].n)))[j]+=g*d;}}}
    J.push(gl[inK]?Array.from(gl[inK]):new Array(I.n).fill(0));}
  return J;
}
let st={};
function compute(){
  forward();if(typeof ALIAS!=='undefined')for(const a in ALIAS)TS[a]=TS[ALIAS[a]];backward();
  const J={};for(const m in MODS)J[m]=MODS[m].io.map(([a,b])=>jac(m,a,b));
  st={loss:TS.loss.v[0],J};
}
const num=(v,d)=>{if(v===-Infinity)return'-∞';const a=Math.abs(v);let s=d!=null?v.toFixed(d):a>=100?v.toFixed(0):a>=10?v.toFixed(1):a>=1?v.toFixed(2):a>=0.01||a===0?v.toFixed(3):v.toExponential(1);if(/^-0\.?0*$/.test(s))s=s.slice(1);return s;};
const fmt=(v,d)=>num(v,d).replace('-','−');
const nt=(v,d)=>v===-Infinity?'-\\infty':num(v,d==null?2:d);
function MX(k,g,d){const t=TS[k],a=g?t.g:t.v,rows=[];for(let r=0;r<t.R;r++)rows.push(range(t.C).map(c=>nt(a[r*t.C+c],d)).join('&'));return `\\begin{pmatrix}${rows.join('\\\\')}\\end{pmatrix}`;}
const MG=(k,d)=>MX(k,true,d==null?3:d);
const EQ=s=>`<div class="eq">$$${s}$$</div>`;
const DL=k=>`\\frac{\\partial\\ell}{\\partial ${TD[k].t.split('=')[0]}}`;
const tokS=i=>`<span class="ctx">${TOK[i]}</span>`;
const CF={},CB={};
function ropeSVG(kind){return '';}
CF.f0=()=>['整张图',tx`<p>这一页从 token id 一直画到损失。embedding 按 id 取出每个位置的向量，中间的 $N-1$ 个 decoder 层收成一个占位盒子，最后一层完整展开：attention 和 SwiGLU MLP 各带一个残差，之后是 final norm、lm_head、log_softmax 和 nll_loss。序列长度 $T=3$，每个激活都是 $T$ 行的板，最前面一行是最后一个位置。输入是 ${tokS(P.seq[0])} ${tokS(P.seq[1])} ${tokS(P.seq[2])}，标签右移一位，是 ${tokS(P.seq[1])} ${tokS(P.seq[2])} ${tokS(P.seq[3])}，损失对三个位置取平均。前向从下往上，反向从上往下。尺寸用 $d=4$、$f=6$、$V=5$，2 个 query head、1 个 kv head，$d_h=4$。点两侧的大括号可以把一个模块折叠起来。</p>`+EQ(tx`\ell=\frac{1}{T}\sum_{t=1}^{T}-\log\operatorname{softmax}(\mathbf{z}_t)_{y_t}=${num(st.loss,3)}`)];
CF.in=()=>['输入和参数',tx`<p>输入只有 token id。其余都是参数：embedding 表 $E$、三个 RMSNorm 的权重、q_norm 和 k_norm 的权重、四个 attention 投影、三个 MLP 投影和 lm_head。Linear 的权重在 PyTorch 里存成 (out, in)，图里一律画成转置后的板，这样每次矩阵乘都是「左边的行乘右边的板」。RoPE 的 cos/sin 表和 causal mask 也在这一步出现，它们由位置决定，不是参数，也不需要梯度。</p>`+EQ(tx`x=(${P.seq.slice(0,T).join(',')}),\qquad y=(${P.seq.slice(1).join(',')})`)];
CF.emb=()=>['embed_tokens',tx`<p>每个位置按自己的 token id 从 $E$ 里取出一行。它等价于 one-hot 向量乘 $E$，但实现里就是按下标取行，所以每个位置只和被取到的那一行有连线。</p>`+EQ(tx`\mathbf{e}_t=E[x_t],\qquad \mathbf{e}=${MX('e')}`)];
CF.F_pre=()=>['⋯ 第 1 到 N−1 层（占位）',tx`<p>真实模型在 embedding 和最后一层之间还有 $N-1$ 个结构完全相同的 Qwen3DecoderLayer，Qwen3-8B 里 $N=36$。这里把它们收成一个盒子，只用来示意：盒子里没有真实计算，数值按恒等映射处理，Jacobian 取单位阵，所以 $\mathbf{x}_0$ 和 $\mathbf{e}$ 相同。它右边的大括号和上面「第 N 层」的大括号是同一级，表示它们是同一种层。这个盒子不能展开。</p>`+EQ(tx`\mathbf{x}_0=\mathrm{layers}_{1..N-1}(\mathbf{e})\;\approx\;\mathbf{e}\quad(\text{示意})`)];
CB.B_pre=()=>['穿过占位的 N−1 层',tx`<p>因为盒子按恒等映射处理，梯度原样传给 $\mathbf{e}$。真实模型里，这里会依次穿过 $N-1$ 个 decoder 层的反向，每一层的结构都和上面展开的这一层相同。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{e}}=\frac{\partial\ell}{\partial\mathbf{x}_0}=${MG('e')}`)];
CB.b_emb=()=>['传到 embedding 表',tx`<p>取行的反向是把梯度加回被取到的那一行（scatter-add）。没有出现在输入里的 token，那一行的梯度是 0；同一个 token 出现几次，它那一行就累加几次。</p>`+EQ(tx`\frac{\partial\ell}{\partial E[v]}=\sum_{t:\,x_t=v}\frac{\partial\ell}{\partial\mathbf{e}_t},\qquad \frac{\partial\ell}{\partial E}=${MG('E')}`)];

function rmsCaps(ids,c){
  const [sq,rs,ah,out]=ids;
  CF[sq]=()=>[`${c.nm}：逐元素平方`,`<p>${c.intro}HF 的实现先把输入转成 float32 再平方。</p>`+EQ(`${c.X}^{2}=${MX(c.sq)}`)];
  CF[rs]=()=>['均方根的倒数',tx`<p>每一行对平方取平均得到 ms，加上 $\epsilon=10^{-6}$ 再取平方根的倒数，得到这一行的 $r$。$T$ 个位置各算各的，互不影响，所以 ms 和 $r$ 都是每行一个。</p>`+EQ(tx`\mathrm{ms}=${MX(c.ms,0,3)},\qquad r=${MX(c.r,0,3)}`)];
  CF[ah]=()=>['归一化',tx`<p>每个分量乘上自己那一行的 $r$。一个元素直接依赖它自己，又经过 $r$ 依赖同一行的其他元素，所以画了两组连线；不同的行之间没有连线。这里不减均值，也没有 bias。</p>`+EQ(`${c.AH}=${MX(c.ah)}`)];
  CF[out]=()=>[`乘上权重 ${c.gs}`,`<p>$${c.GG}$ 是长度为 $d$ 的向量，所有位置共用，所以它的每个分量连到每一行的同一列。${c.outNote||''}</p>`+EQ(`${c.OUT}=${c.GG}\\odot${c.AH}=${MX(c.out)}`)];
  CB['b_'+out]=()=>[`传到 ${c.gs} 和 x̂`,`<p>逐元素乘的两个输入互为对方的局部导数。$${c.GG}$ 被所有位置共用，所以它的梯度要把 $T$ 行加起来。</p>`+EQ(`\\frac{\\partial\\ell}{\\partial${c.GG}}=\\sum_t\\frac{\\partial\\ell}{\\partial${c.OUT}_t}\\odot${c.AH}_t=${MG(c.G)},\\qquad \\mathbf{u}=\\frac{\\partial\\ell}{\\partial${c.OUT}}\\odot${c.GG}=${MG(c.ah)}`)];
  CB['b_'+ah]=()=>['传到 r，以及输入的直连那条','<p>$r_t$ 有 $d$ 条出边，局部导数各是这一行的输入分量，相加得到 $\\partial\\ell/\\partial r_t=\\mathbf{x}_t\\cdot\\mathbf{u}_t$。输入沿直连那条边拿到 $r_t u_{t,k}$。</p>'+EQ(`\\frac{\\partial\\ell}{\\partial r}=${MG(c.r)}`)];
  CB['b_'+rs]=()=>['传到 ms 和平方',tx`<p>$\partial r/\partial\,\mathrm{ms}=-r^3/2$。平均值再把梯度均分给这一行的每个平方。</p>`+EQ(tx`\frac{\partial\ell}{\partial\,\mathrm{ms}}=${MG(c.ms)},\qquad \frac{\partial\ell}{\partial ${c.X}^{2}}=${MG(c.sq)}`)];
  CB['b_'+sq]=()=>['传到输入',`<p>平方这一路的局部导数是 $2x$。两条路相加以后，RMSNorm 交给输入的那部分梯度和这一行的输入垂直，因为 RMSNorm 对整行的缩放不敏感。${c.inNote||''}</p>`+EQ(`\\frac{\\partial\\ell}{\\partial${c.X}}=${MG(c.X)}`)];
}
rmsCaps(['r1sq','r1rs','r1ah','r1n'],{nm:'input_layernorm',intro:'attention 之前先做一次 RMSNorm，也就是 input_layernorm。',X:'\\mathbf{x}_0',sq:'sq1',ms:'ms1',r:'r1',ah:'ah1',AH:'\\hat{\\mathbf{x}}_0',G:'G1',GG:'\\boldsymbol{\\gamma}_1',gs:'γ₁',out:'n1',OUT:'\\mathbf{n}_1',
  outNote:'$\\mathbf{n}_1$ 是 q、k、v 三个投影共同的输入。',inNote:'$\\mathbf{x}_0$ 还会沿残差直接收到 $\\partial\\ell/\\partial\\mathbf{a}$，图里显示的是相加后的总量。'});
rmsCaps(['r2sq','r2rs','r2ah','r2n'],{nm:'post_attention_layernorm',intro:'MLP 之前再做一次 RMSNorm，计算和 input_layernorm 相同。',X:'\\mathbf{a}',sq:'sq2',ms:'ms2',r:'r2',ah:'ah2',AH:'\\hat{\\mathbf{a}}',G:'G2',GG:'\\boldsymbol{\\gamma}_2',gs:'γ₂',out:'n',OUT:'\\mathbf{n}',
  outNote:'$\\mathbf{n}$ 是 MLP 的输入。',inNote:'$\\mathbf{a}$ 还会沿 MLP 的残差直接收到 $\\partial\\ell/\\partial\\mathbf{x}$，图里显示的是相加后的总量。'});
rmsCaps(['fsq','frs','fxh','fh'],{nm:'final norm',intro:'从这里开始是 final norm（<code>model.norm</code>），在所有 decoder 层之后、lm_head 之前。',X:'\\mathbf{x}',sq:'sq',ms:'ms',r:'r',ah:'xh',AH:'\\hat{\\mathbf{x}}',G:'G',GG:'\\boldsymbol{\\gamma}',gs:'γ',out:'h',OUT:'\\mathbf{h}',outNote:'$\\mathbf{h}$ 是 lm_head 的输入。'});

CF.qkv=()=>['q_proj、k_proj、v_proj',tx`<p>三个投影的输入都是 $\mathbf{n}_1$，所以三块权重拼成一块板来画：$n_{t,k}$ 沿第 $k$ 行依次穿过 $W_q^{\top}$、$W_k^{\top}$、$W_v^{\top}$。q 有 2 个 head，每个 head 宽 $d_h=4$，一共宽 8；k 和 v 只有 1 个 kv head，各宽 4，两个 query head 共用它，这就是 GQA。Qwen3 的这三个 Linear 都没有 bias。代码里接下来的 <code>view</code> 只是把列按 head 分组，不移动数据，图里用列之间的空隙表示 head 的分界；q_norm、k_norm 之后的 <code>transpose(1, 2)</code> 也只是换轴。</p>`+EQ(tx`\mathbf{q}=\mathbf{n}_1W_q^{\top}=${MX('q')}`)+EQ(tx`\mathbf{k}=\mathbf{n}_1W_k^{\top}=${MX('k')},\qquad \mathbf{v}=\mathbf{n}_1W_v^{\top}=${MX('v')}`)];
CB.b_qkv=()=>['传到三个投影的权重和 n₁',tx`<p>三块权重的梯度都是「输入的转置乘输出的梯度」，$T$ 个位置的贡献在矩阵乘里自动加起来。$n_{t,k}$ 沿第 $k$ 行穿过三块板，一共 $8+4+4$ 段，全部相加。</p>`+EQ(tx`\frac{\partial\ell}{\partial W_q^{\top}}=\mathbf{n}_1^{\top}\frac{\partial\ell}{\partial\mathbf{q}},\qquad \frac{\partial\ell}{\partial\mathbf{n}_1}=\frac{\partial\ell}{\partial\mathbf{q}}W_q+\frac{\partial\ell}{\partial\mathbf{k}}W_k+\frac{\partial\ell}{\partial\mathbf{v}}W_v=${MG('n1')}`)];

CF.qksq=()=>['q_norm、k_norm：逐元素平方',tx`<p>Qwen3 比 Qwen2 多了这一步：在 RoPE 之前，对 q 和 k 的每个 head 各做一次 RMSNorm，范围是 $d_h$，而不是整个 hidden size。它让点积的尺度更稳定。</p>`+EQ(tx`\mathbf{q}^{2}=${MX('qsq')}`)];
CF.qkrs=()=>['每个 head 的均方根倒数',tx`<p>q 的每个位置、每个 head 各有一个 $r^{q}$，k 只有一个 kv head，每个位置一个 $r^{k}$。</p>`+EQ(tx`r^{q}=${MX('qrr',0,3)},\qquad r^{k}=${MX('krr',0,3)}`)];
CF.qkah=()=>['归一化',tx`<p>每个分量乘上自己那一组的 $r$。连线只在同一个 head 的同一行之内。</p>`+EQ(tx`\hat{\mathbf{q}}=${MX('qah')}`)];
CF.qkn=()=>['乘上 γ_q 和 γ_k',tx`<p>$\boldsymbol{\gamma}_q$ 的长度是 $d_h$，两个 query head 和所有位置共用，所以它的每个分量连到 $2\times T$ 个元素。</p>`+EQ(tx`\bar{\mathbf{q}}=${MX('qn')},\qquad \bar{\mathbf{k}}=${MX('kn')}`)];
CB.b_qkn=()=>['传到 γ_q、γ_k 和 q̂、k̂',tx`<p>$\boldsymbol{\gamma}_q$ 被两个 head 和三个位置共用，它的梯度把这六组贡献加起来。</p>`+EQ(tx`\frac{\partial\ell}{\partial\boldsymbol{\gamma}_q}=${MG('Gq')},\qquad \frac{\partial\ell}{\partial\boldsymbol{\gamma}_k}=${MG('Gk')}`)];
CB.b_qkah=()=>['传到 r 和 q、k 的直连那条',tx`<p>和普通的 RMSNorm 一样，只是每组只有 $d_h$ 个分量。</p>`+EQ(tx`\frac{\partial\ell}{\partial r^{q}}=${MG('qrr')},\qquad \frac{\partial\ell}{\partial r^{k}}=${MG('krr')}`)];
CB.b_qkrs=()=>['传到 ms 和平方',tx`<p>$\partial r/\partial\,\mathrm{ms}=-r^3/2$，再除以 $d_h$ 均分给这一组的每个平方。</p>`+EQ(tx`\frac{\partial\ell}{\partial\,\mathrm{ms}^{q}}=${MG('qms')},\qquad \frac{\partial\ell}{\partial\,\mathrm{ms}^{k}}=${MG('kms')}`)];
CB.b_qksq=()=>['传到 q 和 k',tx`<p>两条路相加。每个 head 交回去的梯度都和这个 head 的输入垂直。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{q}}=${MG('q')}`)+EQ(tx`\frac{\partial\ell}{\partial\mathbf{k}}=${MG('k')}`)];

CF.qkc=()=>['RoPE：乘 cos',tx`<p>RoPE 的公式是 $\mathbf{q}'=\bar{\mathbf{q}}\odot\cos+\mathrm{rotate\_half}(\bar{\mathbf{q}})\odot\sin$，这里按代码的顺序拆成四步。cos/sin 表的每一行对应一个位置；列按频率排，$\omega_i=\theta^{-2i/d_h}$，Qwen3 的 rope_theta 是 $\theta=10^6$；这里只有 $d_h=4$ 和 3 个位置，用 $10^6$ 的话第 2、4 列几乎不转，看不出循环，所以改用 $\theta=10$，频率是 $(1,\,0.316,\,1,\,0.316)$。表的第 1 行是 position_id $t=0$，角度为 0，cos 全是 1；第 2 行 $t=1$：$\cos 1=0.540$、$\cos 0.316=0.950$；第 3 行 $t=2$：$\cos 2=-0.416$、$\cos 0.632=0.807$。第 $c$ 列和第 $c+2$ 列是 rotate_half 配成的一对，HF 用 <code>torch.cat((freqs, freqs))</code> 生成这张表，所以它们的频率相同、数值也相同。两个 q head 和 k 用的是同一张表。下图把位置延长到 24，阴影是这里用到的前 3 个位置：快的那组周期约 6.3 个位置，慢的那组约 20 个位置。</p>`+ropeSVG('cos')+EQ(tx`\cos=${MX('cos',0,3)},\qquad \bar{\mathbf{q}}\odot\cos=${MX('qc')}`)];
CF.rh=()=>['rotate_half',tx`<p>把每个 head 的后半段取负号放到前面，前半段放到后面。它只是重新排列加取负号，所以局部导数是 $\pm1$，每个元素只连一条线。</p>`+EQ(tx`\mathrm{rotate\_half}(x_1,x_2,x_3,x_4)=(-x_3,-x_4,x_1,x_2),\qquad \mathrm{rh}(\bar{\mathbf{q}})=${MX('qrh')}`)];
CF.qks=()=>['乘 sin',tx`<p>第 1 行（$t=0$）的 sin 全是 0，所以第一行在这一步全部变成 0。sin 表同样是周期的，和 cos 差四分之一个周期。</p>`+ropeSVG('sin')+EQ(tx`\sin=${MX('sin',0,3)},\qquad \mathrm{rh}(\bar{\mathbf{q}})\odot\sin=${MX('qs')}`)];
CF.qkr=()=>['相加',tx`<p>合起来看，每个 head 里的第 $c$ 和第 $c+2$ 个分量组成一对，被一起转了角度 $t\,\omega_c$。q 和 k 转过的角度只取决于各自的位置，所以 $\mathbf{q}'_t\cdot\mathbf{k}'_s$ 只依赖位置差 $t-s$。</p>`+EQ(tx`\mathbf{q}'=${MX('qr')},\qquad \mathbf{k}'=${MX('kr')}`)];
CB.b_qkr=()=>['RoPE 反向：穿过相加',tx`<p>相加的两个输入都拿到同样的梯度。</p>`+EQ(tx`\frac{\partial\ell}{\partial(\bar{\mathbf{q}}\odot\cos)}=\frac{\partial\ell}{\partial(\mathrm{rh}\odot\sin)}=\frac{\partial\ell}{\partial\mathbf{q}'}=${MG('qr')}`)];
CB.b_qks=()=>['传到 rotate_half 的输出',tx`<p>局部导数是对应位置的 sin。</p>`+EQ(tx`\frac{\partial\ell}{\partial\,\mathrm{rh}(\bar{\mathbf{q}})}=\frac{\partial\ell}{\partial\mathbf{q}'}\odot\sin=${MG('qrh')}`)];
CB.b_rh=()=>['穿过 rotate_half',tx`<p>反向就是把梯度按相反的方向搬回去：$\bar q_{c}$ 从 $\mathrm{rh}_{c+2}$ 拿到 $+1$ 倍，$\bar q_{c+2}$ 从 $\mathrm{rh}_{c}$ 拿到 $-1$ 倍。</p>`];
CB.b_qkc=()=>['传到 q̄、k̄',tx`<p>$\bar{\mathbf{q}}$ 有两条路：乘 cos 的直连，和经过 rotate_half 的那条。两条相加，等于用 $-t\,\omega$ 把梯度转回去。</p>`+EQ(tx`\frac{\partial\ell}{\partial\bar{\mathbf{q}}}=${MG('qn')},\qquad \frac{\partial\ell}{\partial\bar{\mathbf{k}}}=${MG('kn')}`)];

CF.krep=()=>['repeat_kv：复制 k',tx`<p>GQA 里 kv head 比 query head 少：这里 2 个 query head 共用 1 个 kv head。HF 的 eager 实现先用 <code>repeat_kv</code> 把 k 复制 $n_\text{rep}=2$ 份，每个 query head 拿到一份。复制的局部导数是 1。</p>`+EQ(tx`\mathbf{k}'^{(1)}=\mathbf{k}'^{(2)}=\mathbf{k}'`)];
CF.vrep=()=>['repeat_kv：复制 v',tx`<p>v 也复制两份，直接画成上方 $P\mathbf{v}$ 那一步用的两块板。</p>`+EQ(tx`\mathbf{v}^{(1)}=\mathbf{v}^{(2)}=\mathbf{v}=${MX('v')}`)];
CF.kT=()=>['转置',tx`<p><code>key_states.transpose(2, 3)</code> 把 $d_h$ 换到行、位置换到列，接下来就能按「行向量乘板」来算 $\mathbf{q}'\mathbf{k}'^{\top}$。转置只是换轴，局部导数是 1。</p>`];
CF.S=()=>['打分：q kᵀ',tx`<p>每个 head 各自算一个 $T\times T$ 的分数表：$S_{t,s}$ 是位置 $t$ 的 query 和位置 $s$ 的 key 的点积。$q'_{t,c}$ 沿第 $c$ 行经过 $T$ 个方块，每一列累加后落到 $S_{t,s}$。</p>`+EQ(tx`S^{(1)}=${MX('S0')},\qquad S^{(2)}=${MX('S1')}`)];
CF.Ssc=()=>['缩放',tx`<p>乘上 scaling $=1/\sqrt{d_h}=0.5$。点积的方差随 $d_h$ 增长，除以 $\sqrt{d_h}$ 让 softmax 的输入不至于太大。</p>`+EQ(tx`\tilde{S}^{(1)}=${MX('Ss0')},\qquad \tilde{S}^{(2)}=${MX('Ss1')}`)];
CF.mask=()=>['causal mask',tx`<p>加上 causal mask：$s>t$ 的位置加 $-\infty$，位置 $t$ 只能看到自己和之前的位置。mask 是常量，不需要梯度。HF 实现里填的不是真正的 $-\infty$，而是这个 dtype 能表示的最小值，取 exp 之后同样是 0。</p>`+EQ(tx`M=${MX('M')},\qquad \bar{S}^{(1)}=${MX('Sm0')}`)];
CF.exp=()=>['softmax：取 exp',tx`<p>softmax 按定义拆成三步画：取 exp、按行求和、相除。$e^{-\infty}=0$，被 mask 的位置在这里变成 0。torch 的 softmax kernel 会先减去每行的最大值再取 exp，防止溢出；这一项在相除时会约掉，结果和梯度都不变，这里为了看清结构按定义画。</p>`+EQ(tx`\mathrm{ex}^{(1)}=${MX('Se0',0,3)},\qquad \mathrm{ex}^{(2)}=${MX('Se1',0,3)}`)];
CF.sum=()=>['按行求和',tx`<p>每个 head、每一行各有一个 $Z$。</p>`+EQ(tx`Z^{(1)}=${MX('Z0',0,3)},\qquad Z^{(2)}=${MX('Z1',0,3)}`)];
CF.div=()=>['注意力权重',tx`<p>每一行除以自己的 $Z$，得到位置 $t$ 对之前各位置的注意力分布，每行加起来是 1。第一行只能看自己，所以 $P_{1,1}=1$。Qwen3 的 attention_dropout 是 0，这里没有 dropout。</p>`+EQ(tx`P^{(1)}=${MX('P0',0,3)},\qquad P^{(2)}=${MX('P1',0,3)}`)];
CF.PV=()=>['P v',tx`<p>每个 head 用注意力权重对 $\mathbf{v}$ 的各行做加权平均。板的行是位置 $s$，列是 $d_h$；$P_{t,s}$ 沿第 $s$ 行经过 $d_h$ 个方块。</p>`+EQ(tx`\mathbf{o}^{(1)}=P^{(1)}\mathbf{v}^{(1)}=${MX('O0')},\qquad \mathbf{o}^{(2)}=${MX('O1')}`)];
CF.cat=()=>['拼回 T × 8',tx`<p><code>attn_output.transpose(1, 2).reshape(B, T, -1)</code> 把两个 head 的输出并排拼回去。只是换轴和重排，局部导数是 1。</p>`+EQ(tx`\mathbf{o}=[\,\mathbf{o}^{(1)}\,|\,\mathbf{o}^{(2)}\,]=${MX('O')}`)];
CF.oproj=()=>['o_proj',tx`<p>投影回 $d$ 维。o_proj 也没有 bias。</p>`+EQ(tx`\mathbf{att}=\mathbf{o}W_o^{\top}=${MX('ao')}\qquad (T\times 8)(8\times d)=(T\times d)`)];
CF.res1=()=>['残差相加',tx`<p>attention 算出来的只是一个增量，加回 residual stream。$\mathbf{x}_0$ 到 $\mathbf{a}$ 的那组长弧线就是残差连接，它绕过了整个 attention。</p>`+EQ(tx`\mathbf{a}=\mathbf{x}_0+\mathbf{att}=${MX('a')}`)];
CB.b_res1=()=>['穿过 attention 的残差',tx`<p>相加对两个输入的局部导数都是 1，梯度复制成两份：一份给 $\mathbf{att}$，进入 attention；一份沿残差直接交给 $\mathbf{x}_0$。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{att}}=\frac{\partial\ell}{\partial\mathbf{a}}=${MG('a')}`)];
CB.b_oproj=()=>['传到 o_proj 权重和 o',tx`<p>和其他 Linear 一样：权重的梯度是外积，$T$ 个位置的贡献加在一起；输入的梯度是输出的梯度乘权重。</p>`+EQ(tx`\frac{\partial\ell}{\partial W_o^{\top}}=\mathbf{o}^{\top}\frac{\partial\ell}{\partial\mathbf{att}},\qquad \frac{\partial\ell}{\partial\mathbf{o}}=\frac{\partial\ell}{\partial\mathbf{att}}W_o=${MG('O')}`)];
CB.b_cat=()=>['拆回两个 head',tx`<p>拼接的反向是把梯度按列切回去。</p>`];
CB.b_PV=()=>['传到 P 和 v',tx`<p>$\partial\ell/\partial P=\frac{\partial\ell}{\partial\mathbf{o}}\mathbf{v}^{\top}$，$\partial\ell/\partial\mathbf{v}=P^{\top}\frac{\partial\ell}{\partial\mathbf{o}}$。被 mask 的格子 $P_{t,s}=0$，所以它们不会把梯度传给后面位置的 v。</p>`+EQ(tx`\frac{\partial\ell}{\partial P^{(1)}}=${MG('P0')},\qquad \frac{\partial\ell}{\partial\mathbf{v}^{(1)}}=${MG('V0')}`)];
CB.b_div=()=>['softmax 反向：穿过相除',tx`<p>$\mathrm{ex}_{t,s}$ 直连 $P_{t,s}$，局部导数 $1/Z_t$；同时 $Z_t$ 连到这一行每个 $P$，局部导数 $-\mathrm{ex}_{t,s}/Z_t^2$。</p>`+EQ(tx`\frac{\partial\ell}{\partial Z^{(1)}}=${MG('Z0')}`)];
CB.b_sum=()=>['传回 ex',tx`<p>求和的局部导数是 1，$Z_t$ 的梯度复制给这一行的每个 $\mathrm{ex}$，再和直连那条加起来。</p>`+EQ(tx`\frac{\partial\ell}{\partial\,\mathrm{ex}^{(1)}}=${MG('Se0')}`)];
CB.b_exp=()=>['穿过 exp',tx`<p>exp 的导数是它自己。合起来就是熟悉的 softmax 反向：$\frac{\partial\ell}{\partial\bar S_{t,s}}=P_{t,s}\big(g_{t,s}-\sum_{s'}P_{t,s'}g_{t,s'}\big)$，其中 $g=\partial\ell/\partial P$。被 mask 的格子 ex 是 0，梯度也是 0。</p>`+EQ(tx`\frac{\partial\ell}{\partial\bar{S}^{(1)}}=${MG('Sm0')},\qquad \frac{\partial\ell}{\partial\bar{S}^{(2)}}=${MG('Sm1')}`)];
CB.b_mask=()=>['穿过 mask',tx`<p>加常量，梯度原样传下去。mask 本身不需要梯度。</p>`];
CB.b_Ssc=()=>['穿过缩放',tx`<p>局部导数是 $1/\sqrt{d_h}=0.5$。</p>`+EQ(tx`\frac{\partial\ell}{\partial S^{(1)}}=${MG('S0')}`)];
CB.b_S=()=>['传到 q′ 和 kᵀ',tx`<p>和普通矩阵乘一样，只是这里两边都是激活：$\frac{\partial\ell}{\partial\mathbf{q}'}=\frac{\partial\ell}{\partial S}\mathbf{k}'$，$\frac{\partial\ell}{\partial\mathbf{k}'^{\top}}=\mathbf{q}'^{\top}\frac{\partial\ell}{\partial S}$。q′ 的两个 head 各收到自己那一份。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{q}'}=${MG('qr')}`)];
CB.b_kT=()=>['转置回去',tx`<p>把梯度转置回 $T\times d_h$。</p>`];
CB.b_krep=()=>['repeat_kv 反向：两份加回一份',tx`<p>k 被复制给了两个 query head，反向时两份梯度在这里相加，回到唯一的 kv head。GQA 省下的是 k、v 的投影和缓存，梯度的数学形式不变。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{k}'}=\frac{\partial\ell}{\partial\mathbf{k}'^{(1)}}+\frac{\partial\ell}{\partial\mathbf{k}'^{(2)}}=${MG('kr')}`)];
CB.b_vrep=()=>['v 的两份加回一份',tx`<p>v 同样把两个 head 的梯度相加。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{v}}=${MG('v')}`)];

CF.gu=()=>['gate_proj 和 up_proj',tx`<p>两个投影的输入都是 $\mathbf{n}$，两块权重拼在一起画，$n_{t,k}$ 沿第 $k$ 行一次穿过两块。维度从 $d$ 扩大到 $f$，Qwen3-8B 里是 4096 到 12288。</p>`+EQ(tx`\mathbf{gate}=\mathbf{n}W_g^{\top}=${MX('gate')}`)];
CF.silu=()=>['SiLU',tx`<p>gate 逐元素经过 $\mathrm{SiLU}(v)=v\,\sigma(v)$。正数基本原样通过，负数被压到接近 0。</p>`+EQ(tx`\mathbf{s}=${MX('s')}`)];
CF.mul=()=>['门控相乘',tx`<p>$\mathbf{s}$ 和 $\mathbf{up}$ 逐元素相乘，$\mathbf{s}$ 决定 $\mathbf{up}$ 的每个分量放行多少。</p>`+EQ(tx`\mathbf{m}=${MX('m')}`)];
CF.dn=()=>['down_proj',tx`<p>再投影回 $d$ 维，摆法和 lm_head 一样。</p>`+EQ(tx`\mathbf{down}=\mathbf{m}W_d^{\top}=${MX('down')}`)];
CF.res2=()=>['残差相加',tx`<p>MLP 的输出加回 residual stream，得到这一层的输出。</p>`+EQ(tx`\mathbf{x}=\mathbf{a}+\mathbf{down}=${MX('x')}`)];
CB.b_res2=()=>['穿过 MLP 的残差',tx`<p>梯度复制成两份，一份进入 MLP，一份沿残差直接交给 $\mathbf{a}$。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{down}}=\frac{\partial\ell}{\partial\mathbf{x}}=${MG('x')}`)];
CB.b_dn=()=>['传到 down_proj 权重和 m',tx`<p>权重的梯度是 $\mathbf{m}^{\top}\frac{\partial\ell}{\partial\mathbf{down}}$，三个位置加在一起。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{m}}=${MG('m')}`)];
CB.b_mul=()=>['传到 s 和 up',tx`<p>逐元素乘，两个输入互为对方的局部导数。门 $s$ 接近 0 的地方，up 那一路几乎收不到梯度。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{up}}=${MG('up')}`)];
CB.b_silu=()=>['穿过 SiLU',tx`<p>SiLU 的导数是 $\sigma(v)\big(1+v(1-\sigma(v))\big)$。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{gate}}=${MG('gate')}`)];
CB.b_gu=()=>['传到 gate_proj、up_proj 权重和 n',tx`<p>$n_{t,k}$ 那一行同时穿过 gate 和 up 两块，一共 $2f$ 段，全部相加。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{n}}=${MG('n')}`)];

CF.lm=()=>['lm_head',tx`<p>每个位置的 $\mathbf{h}_t$ 都乘同一块 $W^{\top}$，得到这个位置对下一个 token 的 logits。$h_{t,k}$ 沿第 $k$ 行经过每个方块，每一列累加后落到 $z_{t,i}$。</p>`+EQ(tx`\mathbf{z}=\mathbf{h}W^{\top}=${MX('z')}`)];
CF.lse=()=>['log-sum-exp',tx`<p>每个位置各算一个 LSE。实现里先减去这一行的最大值再取指数，不会溢出。</p>`+EQ(tx`\mathrm{LSE}_t=\log\sum_i e^{z_{t,i}}=${MX('L',0,3)}`)];
CF.lp=()=>['log_softmax',tx`<p>每个 logit 减去自己那一行的 LSE。</p>`+EQ(tx`\log\mathbf{p}=${MX('lp')}`)];
CF.nll=()=>['nll_loss（逐位置）',`<p>每个位置按标签取出目标 token 的 log p，取负号。标签是输入右移一位：位置 1 的标签是 ${tokS(P.seq[1])}，位置 2 是 ${tokS(P.seq[2])}，位置 3 是 ${tokS(P.seq[3])}。每一行只有标签那一格有连线。</p>`+EQ(tx`\ell_t=-\log p_{t,y_t}:\quad ${MX('lt',0,3)}`)];
CF.mean=()=>['取平均',tx`<p><code>F.nll_loss</code> 默认 <code>reduction='mean'</code>，对 $T$ 个位置取平均。它在一个算子里同时做了取格、取负和平均，这里分成两步画。</p>`+EQ(tx`\ell=\frac{1}{T}\sum_t\ell_t=${num(st.loss,3)}`)];
CB.b_mean=()=>['从损失出发',tx`<p>平均值把梯度 $1/T$ 分给每个位置。</p>`+EQ(tx`\frac{\partial\ell}{\partial\ell_t}=\frac{1}{T}=${num(1/T,3)}`)];
CB.b_nll=()=>['穿过 nll',tx`<p>每一行只有标签那一格收到 $-1/T$，其余是 0。</p>`+EQ(tx`\frac{\partial\ell}{\partial\log\mathbf{p}}=${MG('lp')}`)];
CB.b_lp=()=>['传到 logits 和 LSE',tx`<p>每行的 LSE 收到 $1/T$。$z$ 的直连那条只有标签格非零。</p>`+EQ(tx`\frac{\partial\ell}{\partial\,\mathrm{LSE}}=${MG('L')}`)];
CB.b_lse=()=>['穿过 LSE',tx`<p>两条路加起来，每一行都是 $(\mathbf{p}_t-\mathbf{e}_{y_t})/T$。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{z}}=${MG('z')}`)];
CB.b_lm=()=>['传到 lm_head 权重和 h',tx`<p>$\partial\ell/\partial W^{\top}=\mathbf{h}^{\top}\frac{\partial\ell}{\partial\mathbf{z}}$，三个位置的外积加在一起。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{h}}=\frac{\partial\ell}{\partial\mathbf{z}}W=${MG('h')}`)];

const JN=(m,q)=>{const io=MODS[m].io[q||0];return [io[0],io[1]];};
CF.F_attn=()=>['self_attn（已折叠）',tx`<p>q/k/v 投影、q_norm/k_norm、RoPE、eager_attention_forward 和 o_proj 都收进了盒子，从 $\mathbf{n}_1$ 直接到 $\mathbf{att}$。这和 HF 里 <code>self.self_attn(hidden_states)</code> 这一次调用一一对应：input_layernorm 在它之前，残差相加在它之后，都不属于 self_attn，所以仍然画在盒子外面。参数留在盒子里照常显示。连线是整体 Jacobian 的元素：位置 $t$ 的输出只依赖 $s\le t$ 的输入，所以是块下三角。</p>`+EQ(tx`\mathbf{att}=\mathrm{self\_attn}(\mathbf{n}_1)=${MX('ao')}`)];
CB.B_attn=()=>['穿过 self_attn（已折叠）',tx`<p>一次反向同时算出盒子里所有参数的梯度，以及交给 $\mathbf{n}_1$ 的梯度 $\frac{\partial\ell}{\partial\mathbf{att}}J_{\mathrm{attn}}$。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{n}_1}=${MG('n1')}`)];
CF.F_rms1=()=>['input_layernorm（已折叠）',tx`<p>平方、均方根、归一化收在盒子里，$\boldsymbol{\gamma}_1$ 仍然显示。每一行只和自己那一行有关，所以整体 Jacobian 是 $T$ 个 $d\times d$ 的对角块。</p>`+EQ(tx`\mathbf{n}_1=${MX('n1')}`)];
CB.B_rms1=()=>['穿过 input_layernorm（已折叠）',tx`<p>一步算出 $\boldsymbol{\gamma}_1$ 的梯度，以及沿这一路交给 $\mathbf{x}_0$ 的梯度。</p>`+EQ(tx`\frac{\partial\ell}{\partial\boldsymbol{\gamma}_1}=${MG('G1')}`)];
CF.F_qkn=()=>['q_norm 和 k_norm（已折叠）',tx`<p>每个位置、每个 head 各自做 RMSNorm，Jacobian 是 $d_h\times d_h$ 的小块。</p>`+EQ(tx`\bar{\mathbf{q}}=${MX('qn')}`)];
CB.B_qkn=()=>['穿过 q_norm 和 k_norm（已折叠）',''+EQ(tx`\frac{\partial\ell}{\partial\mathbf{q}}=${MG('q')},\qquad \frac{\partial\ell}{\partial\boldsymbol{\gamma}_q}=${MG('Gq')}`)];
CF.F_rope=()=>['RoPE（已折叠）',tx`<p>四步合成一个旋转：每个 head 里的第 $c$ 和第 $c+2$ 个分量组成一对，转过角度 $t\,\omega_c$。Jacobian 是一个个 $2\times2$ 的旋转矩阵，位置 1 是单位阵。</p>`+EQ(tx`\begin{pmatrix}q'_c\\ q'_{c+2}\end{pmatrix}=\begin{pmatrix}\cos t\omega_c & -\sin t\omega_c\\ \sin t\omega_c & \cos t\omega_c\end{pmatrix}\begin{pmatrix}\bar q_c\\ \bar q_{c+2}\end{pmatrix}`)];
CB.B_rope=()=>['穿过 RoPE（已折叠）',tx`<p>旋转矩阵的转置就是它的逆，反向把梯度转回 $-t\,\omega_c$。</p>`+EQ(tx`\frac{\partial\ell}{\partial\bar{\mathbf{q}}}=${MG('qn')}`)];
CF.F_sm=()=>['softmax（已折叠）',tx`<p>每一行的 Jacobian 是 $\mathrm{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^{\top}$，被 mask 的位置那一列和那一行都是 0。</p>`+EQ(tx`P^{(1)}=${MX('P0',0,3)},\qquad P^{(2)}=${MX('P1',0,3)}`)];
CB.B_sm=()=>['穿过 softmax（已折叠）',tx`<p>PyTorch 的 softmax 反向只需要前向保存的输出 $P$：$\frac{\partial\ell}{\partial\bar S}=P\odot\big(g-(g\odot P)\mathbf{1}\big)$。</p>`+EQ(tx`\frac{\partial\ell}{\partial\bar{S}^{(1)}}=${MG('Sm0')}`)];
CF.F_mlp=()=>['mlp（已折叠）',tx`<p>gate_proj、up_proj、SiLU、门控相乘和 down_proj 收进了盒子，对应 HF 里的 <code>self.mlp(hidden_states)</code>。post_attention_layernorm 和残差都在盒子外面。MLP 对每个位置单独计算，所以 Jacobian 是 $T$ 个 $d\times d$ 的对角块。</p>`+EQ(tx`\mathbf{down}=\mathrm{mlp}(\mathbf{n})=${MX('down')}`)];
CB.B_mlp=()=>['穿过 mlp（已折叠）',''+EQ(tx`\frac{\partial\ell}{\partial\mathbf{n}}=\frac{\partial\ell}{\partial\mathbf{down}}J_{\mathrm{mlp}}=${MG('n')}`)];
CF.F_dec=()=>['Qwen3DecoderLayer（已折叠）',tx`<p>整个 decoder 层收成一个盒子，从 $\mathbf{x}_0$ 直接到 $\mathbf{x}$，对应 <code>model.layers[-1](hidden_states)</code>。它和下面的「第 1 到 N−1 层」是同一种结构，只是这一层的计算是真实的。连线是整层的 Jacobian：位置 $t$ 的输出依赖所有 $s\le t$ 的输入，对角块里的单位阵来自两条残差。</p>`+EQ(tx`\mathbf{x}=\mathrm{layer}(\mathbf{x}_0)=${MX('x')}`)];
CB.B_dec=()=>['穿过 Qwen3DecoderLayer（已折叠）',tx`<p>一次反向算出这一层所有参数的梯度，以及交给 $\mathbf{x}_0$ 的梯度。训练时如果开了 gradient checkpointing，这里会先用保存的 $\mathbf{x}_0$ 把整层前向重算一遍，再做反向。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{x}_0}=${MG('x0')}`)];
CF.F_rms2=()=>['post_attention_layernorm（已折叠）',''+EQ(tx`\mathbf{n}=${MX('n')}`)];
CB.B_rms2=()=>['穿过 post_attention_layernorm（已折叠）',''+EQ(tx`\frac{\partial\ell}{\partial\boldsymbol{\gamma}_2}=${MG('G2')}`)];
CF.F_rms=()=>['final norm（已折叠）',''+EQ(tx`\mathbf{h}=${MX('h')}`)];
CB.B_rms=()=>['穿过 final norm（已折叠）',''+EQ(tx`\frac{\partial\ell}{\partial\mathbf{x}}=${MG('x')}`)];
CF.F_ls=()=>['log_softmax（已折叠）',tx`<p>每一行的 Jacobian 是 $\delta_{ij}-p_{t,i}$，行与行之间没有连线。</p>`+EQ(tx`\log\mathbf{p}=${MX('lp')}`)];
CB.B_ls=()=>['穿过 log_softmax（已折叠）',''+EQ(tx`\frac{\partial\ell}{\partial\mathbf{z}}=\frac{\mathbf{p}-\mathbf{e}_{y}}{T}=${MG('z')}`)];
CB.ball=()=>['反向总览',tx`<p>整条反向路径一起显示。损失先把 $1/T$ 分给每个位置；lm_head 之后每个位置各自往下传，直到 attention 才发生跨位置的交流：位置 $t$ 的梯度经过 $P$ 和 $\mathbf{v}$、$\mathbf{k}$ 传给所有 $s\le t$ 的位置。两个残差让 $\mathbf{x}_0$ 至少拿到 $\partial\ell/\partial\mathbf{a}$，再穿过占位的 $N-1$ 层，最后 scatter-add 回 embedding 表里被用到的那几行。</p>`+EQ(tx`\frac{\partial\ell}{\partial\mathbf{x}_0}=${MG('x0')}`)];
if (seq !== undefined) P.seq = seq.slice();
compute();
return { TS, OPS, MODS, ALIAS, st, jac, P, DEF, TOK, T, CF, CB };
}
