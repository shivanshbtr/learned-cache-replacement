"use strict";
(function(){
/* Switching, shared cache, results screens and the router. Uses the helpers exported by index.html as window.AC. */
const {APPS,APPNAME,M,X,$,h,tip,select,seg,table,panel,fmt,pct,spct,css,appColor,BS,BF,loadWin,Grid,playbar,csvRows,parsePct,winHint}=window.AC;
const A=window.AC;

/* ---------- generic CSV table ---------- */
function csvTable(name,{fill=[],bars=false,hl=null,numFrom=1,pol=[]}={}){
  const rows=csvRows(name).map(r=>r.slice());if(!rows.length)return h("p",{class:"note"},"No data for "+name);
  const head=rows.shift();const last=[];
  rows.forEach(r=>{fill.forEach(i=>{if(r[i]===""||r[i]===undefined)r[i]=last[i];else last[i]=r[i];});pol.forEach(i=>{r[i]=polName(r[i]);});});
  const body=rows.map(r=>r.map((c,i)=>{if(bars&&i>=numFrom){const v=parsePct(c);if(v!==null&&/^-?[\d.]+%$/.test(c.trim()))return h("span",{},c.replace(/^-/,"−"),h("span",{class:"bar"+(v<0?" neg":""),style:"width:"+Math.min(100,Math.abs(v)*70)+"px"}));}
    return String(c).replace(/^-(?=\d)/,"−");}));
  return table(head,body,{numFrom,hl});}
const polName=n=>{
  if(!n)return n;
  const base={lru:"LRU",srrip:"SRRIP",drrip:"DRRIP",bip:"BIP",mru:"MRU",belady:"Belady (oracle)",fifo:"FIFO",random:"Random"};if(base[n])return base[n];
  const m=/^(req|own|gen):(mix1b|mix1|mix2)?(spec|allbig|all)_(mlp|gbdt|lstm)$/.exec(n);if(!m)return n;
  const fam=m[4].toUpperCase(),ver=m[2]?" ("+m[2]+")":" (frozen)";
  const what=m[1]==="gen"?(m[3]==="allbig"?"generic-big "+fam:"generic "+fam):(m[1]==="req"?"requester rule, per-app "+fam:"ownership rule, per-app "+fam);
  return what+ver;};
const SIMPLE=["lru","srrip","drrip","bip","mru"];

/* ---------- SWITCHING (E7) ---------- */
const S2={slice:null,mode:"resident",lat:"100"};
function renderSwitching(el){
  el.innerHTML="";
  const keys=Object.keys(M.e7);if(!S2.slice)S2.slice=keys[0];
  const E=M.e7[S2.slice];
  el.append(h("h2",{},"Switching between applications"),
    h("p",{class:"lede"},"The four applications take turns on one cache, a few thousand to millions of instructions at a time. Each application brings its own model. After every switch the other applications have pushed the cache contents around, so the new owner starts cold."));
  el.append(h("div",{class:"row panel"},select("Time slice (instructions per turn)",keys.map(k=>[k,fmt(M.e7[k].slice_instr)+" instr, "+fmt(M.e7[k].switches)+" switches"]),S2.slice,v=>{S2.slice=v;renderSwitching(el);})));
  // results table
  const R=E.runs,lru=R.lru.misses,names=Object.keys(R);
  const best=SIMPLE.reduce((a,b)=>R[b].misses<R[a].misses?b:a);
  const rows=names.map(n=>[polName(n),fmt(R[n].misses),R[n].mpki.toFixed(2),spct((lru-R[n].misses)/lru),n===best?"best simple":spct((R[best].misses-R[n].misses)/R[best].misses)]);
  el.append(panel("Misses over the whole switching run",BF(),
    h("p",{class:"note"},"Models were trained on single applications. “vs best simple”: positive means fewer misses than "+best.toUpperCase()+". Belady is the offline optimum."),
    table(["Policy","Misses","MPKI","vs LRU","vs best simple"],rows,{})));
  // timeline chart: cumulative misses per slice
  const tl=["lru","belady","req:spec_mlp","gen:all_mlp"].filter(k=>R[k]&&R[k].slices);
  const svgW=720,svgH=230,m={l:56,r:120,t:10,b:30};const N=R.lru.slices.length;
  const cum=k=>{let s=0;return R[k].slices.map(x=>s+=x[2]);};
  const C=Object.fromEntries(tl.map(k=>[k,cum(k)]));const ymax=Math.max(...tl.map(k=>C[k][N-1]));
  const sty={lru:["var(--muted)","",""],belady:["var(--ink)","5 4",""],"req:spec_mlp":["var(--graph)","",""],"gen:all_mlp":["var(--hash)","",""]};
  const lab={lru:"LRU",belady:"Belady","req:spec_mlp":"per-app MLP","gen:all_mlp":"generic MLP"};
  const X_=i=>m.l+(svgW-m.l-m.r)*i/(N-1),Y_=v=>m.t+(svgH-m.t-m.b)*(1-v/ymax);
  const step=Math.max(1,Math.floor(N/360));
  let sv=`<svg class="svgc" viewBox="0 0 ${svgW} ${svgH}" width="100%" role="img" aria-label="Cumulative misses over time slices"><line class="ax" x1="${m.l}" y1="${svgH-m.b}" x2="${svgW-m.r}" y2="${svgH-m.b}"/><line class="ax" x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${svgH-m.b}"/>`;
  [0,.5,1].forEach(f=>{const v=ymax*f;sv+=`<text x="${m.l-6}" y="${Y_(v)+4}" text-anchor="end">${fmt(v/1000)}k</text>`;});
  sv+=`<text x="${m.l}" y="${svgH-8}">slice 0</text><text x="${svgW-m.r}" y="${svgH-8}" text-anchor="end">slice ${fmt(N)}</text><text x="${m.l+6}" y="${m.t+10}">cumulative misses</text>`;
  const endY=[];
  tl.forEach(k=>{const pts=[];for(let i=0;i<N;i+=step)pts.push(X_(i).toFixed(1)+","+Y_(C[k][i]).toFixed(1));pts.push(X_(N-1).toFixed(1)+","+Y_(C[k][N-1]).toFixed(1));
    sv+=`<polyline points="${pts.join(" ")}" fill="none" stroke="${sty[k][0]}" stroke-width="${k==="lru"?5:2}" ${sty[k][1]?`stroke-dasharray="${sty[k][1]}"`:""}/>`;endY.push([k,Y_(C[k][N-1])]);});
  endY.sort((a,b)=>a[1]-b[1]);let py=-99;endY.forEach(([k,y])=>{y=Math.max(y,py+13);py=y;sv+=`<text x="${svgW-m.r+6}" y="${y+4}" style="fill:${sty[k][0]==="var(--muted)"?"var(--ink2)":sty[k][0]}">${lab[k]}</text>`;});
  sv+="</svg>";
  // strip of active applications (first slices)
  const nstrip=Math.min(E.runs.lru.slices.length,E.switches>100?60:E.runs.lru.slices.length);
  let st=`<svg class="svgc" viewBox="0 0 ${svgW} 34" width="100%" role="img" aria-label="Which application is active in the first ${nstrip} slices">`;
  const bw=(svgW-m.l-m.r+60)/nstrip;
  for(let i=0;i<nstrip;i++){const a=R.lru.slices[i][0];st+=`<rect x="${m.l+i*bw}" y="4" width="${Math.max(1,bw-1)}" height="18" fill="${appColor(a)}"><title>slice ${i}: ${APPS[a]}</title></rect>`;}
  st+=`<text x="${m.l}" y="32">first ${nstrip} slices: colour = active application, its model is switched in at each colour change</text></svg>`;
  el.append(panel("Misses build up over time",BF(),h("div",{html:sv}),h("div",{html:st,style:"margin-top:6px"}),h("p",{class:"note"},"LRU (grey) and the generic MLP lie almost on top of each other: the generic model saves about as many misses as LRU here.")));
  // interference
  const ir=csvRows("e7_interference").map(r=>r.slice());const hd=ir.shift();const lastv=[];
  ir.forEach(r=>{[0,1].forEach(i=>{if(r[i]==="")r[i]=lastv[i];else lastv[i]=r[i];});});
  const ours=ir.filter(r=>+r[0]===E.slice_instr).map(r=>{const q=r.slice(1);q[0]=polName(q[0]);return q;});
  el.append(panel("Cold-cache effect after a switch",BF(),
    h("p",{class:"note"},"MPKI when the application runs alone vs when it shares the cache in turns. The last two columns compare the miss rate in the first 2,000 accesses after switching in with the overall miss rate."),
    ours.length?table(hd.slice(1),ours,{numFrom:2}):h("p",{class:"note"},"No interference table for this slice length.")));
  // cost model
  const cmr=csvRows("e7_cost_model").slice(1).filter(r=>+r[0]===E.slice_instr&&r[2]===S2.mode&&r[3]===S2.lat);
  const cm=M.cost_model.main;
  el.append(panel("Does it pay off? Cost model",h("span",{class:"pill"},"assumptions, not measurements"),
    h("p",{class:"note"},"Net time = miss time saved against the best simple policy ("+cm.t_miss_ns+" ns per miss) minus the stall for model decisions and activation. Model loads cost "+cm.t_fixed_us+" µs fixed plus size over "+cm.bandwidth_gbps+" GB/s. Real hardware cannot be measured in a simulator, so these parameters are swept in the tables below."),
    h("div",{class:"row"},select("Model placement",[["resident","all models stay loaded"],["on_demand","load the model at each switch"]],S2.mode,v=>{S2.mode=v;renderSwitching(el);}),
      select("Decision time per eviction",[["20","20 ns (hardware-like)"],["100","100 ns"],["1000","1,000 ns"]],S2.lat,v=>{S2.lat=v;renderSwitching(el);})),
    table(["Policy","Misses saved vs best simple","Activation (ms)","Decision stall (ms)","Miss time saved (ms)","Net (ms)"],
      cmr.map(r=>[polName(r[1]),fmt(+r[4]),(+r[6]).toFixed(1),(+r[7]).toFixed(1),(+r[8]).toFixed(1),(+r[9]).toFixed(1).replace("-","−")]),{hl:null}),
    h("p",{class:"note"},"Negative net means the model costs more than it saves. Here the models save no misses at all versus the best simple policy, so no assumption makes them pay off. Training on mixed streams (next screen) changes the miss side of this picture.")));
  el.append(panel("Break-even and activation tables",BF(),
    h("h3",{style:"font-size:14px;margin:6px 0"},"Per-switch model load time (µs) by bandwidth and fixed cost"),csvTable("e6_activation_summary"),
    h("h3",{style:"font-size:14px;margin:14px 0 6px"},"Break-even slice length (instructions between switches)"),csvTable("e7_breakeven",{pol:[0]})));
}

/* ---------- SHARED ---------- */
const S3={val:{tag:"default",family:"mlp",ver:"mix1b",rule:"gen"},test:{set:"test",tag:"default",family:"mlp",ver:"mix1b",rule:"gen"},anim:"gen: mix1b MLP"};
const VERS=[["frozen","frozen (single-app training)"],["mix1","mix1"],["mix1b","mix1b"],["mix2","mix2"]];
const RULES=[["req","req","requester: its own model may evict any line"],["own","own","ownership: a model may only evict its own application's lines"],["gen","gen","generic: one model for everyone, any line"]];
function runOf(S,rule,fam,ver){const scope=rule==="gen"?"gen":"spec";return S.runs.find(r=>r.rule===rule&&r.family===fam&&r.version===ver&&r.scope===scope);}
function heatCol(v){if(v==null)return "transparent";const a=Math.min(.55,Math.abs(v)*2.2);return v>=0?`color-mix(in srgb,var(--pos) ${a*100}%,transparent)`:`color-mix(in srgb,var(--neg) ${a*100}%,transparent)`;}
function sharedBlock(kind,title,badge,note){
  const st=S3[kind==="val"?"val":"test"];const box=h("div");
  const draw=()=>{
    box.innerHTML="";
    const setName=kind==="val"?"val":st.set;const S=X.sets[setName][st.tag];
    const ctl=h("div",{class:"row"});
    if(kind!=="val")ctl.append(h("label",{class:"f"},"Held-out mixture",seg([["test","test mixture 1"],["test2","test mixture 2"]],st.set,v=>{st.set=v;draw();})));
    ctl.append(h("label",{class:"f"},"Cache size",seg([["default","2 MB"],["llc8m","8 MB"]],st.tag,v=>{st.tag=v;draw();})),
      h("label",{class:"f"},"Model family",seg([["mlp","MLP"],["gbdt","GBDT"],["lstm","LSTM"]],st.family,v=>{st.family=v;draw();})));
    box.append(ctl);
    const bsn=S.best_simple,best=S.runs.find(r=>r.name===bsn);
    const lru=S.runs.find(r=>r.name==="lru"),bel=S.runs.find(r=>r.name==="belady");
    // heat grid: rule x version
    const g=h("div",{class:"heat",style:"margin-top:12px"});g.append(h("div",{class:"h"},""));VERS.forEach(([v,t])=>g.append(h("div",{class:"h"},t)));
    RULES.forEach(([rule,nm,desc])=>{g.append(h("div",{class:"h",title:desc},nm+" rule"));
      VERS.forEach(([v])=>{const r=runOf(S,rule,st.family,v);
        if(!r){g.append(h("div",{class:"cell",style:"color:var(--muted);cursor:default;border:1px dashed var(--rule2)"},"not run"));return;}
        const on=st.rule===rule&&st.ver===v;
        g.append(h("button",{class:"cell","aria-pressed":on,style:"background:"+heatCol(r.vs_best)+";color:var(--ink);font:inherit",onclick:()=>{st.rule=rule;st.ver=v;draw();}},
          h("b",{class:"num"},spct(r.vs_best)),h("span",{class:"note num"},"gap "+pct(r.gap,0))));});});
    box.append(h("p",{class:"note",style:"margin:12px 0 4px"},"Each cell: change in misses vs the best simple policy ("+bsn.toUpperCase()+"; + means fewer misses) and the share of the LRU-to-Belady gap closed. Click a cell for details. Rules: req = requester's own model may evict any line, own = a model may only evict its own application's lines, gen = one generic model."));
    box.append(g);
    // details
    const sel=runOf(S,st.rule,st.family,st.ver)||null;
    if(sel){
      const items=[["lru","LRU"],["srrip","SRRIP"],["drrip","DRRIP"],["bip","BIP"],["mru","MRU"]].map(([k,t])=>[t,S.runs.find(r=>r.name===k).misses,"b"]);
      items.push([st.rule+" "+st.family.toUpperCase()+" "+st.ver,sel.misses,"s"],["Belady (oracle)",bel.misses,"o"]);
      items.sort((a,b)=>b[1]-a[1]);
      const mx=Math.max(...items.map(i=>i[1])),rowH=24;
      let sv=`<svg class="svgc" viewBox="0 0 480 ${items.length*rowH+8}" width="100%" role="img" aria-label="Misses by policy">`;
      items.forEach(([t,v,k],i)=>{const y=4+i*rowH,w=210*v/mx;
        sv+=`<text x="150" y="${y+15}" text-anchor="end" ${k==="s"?'style="font-weight:700;fill:var(--ink)"':""}>${A.esc(t)}</text><rect x="158" y="${y+3}" width="${w}" height="16" rx="3" fill="${k==="s"?"var(--ink)":k==="o"?"none":"var(--rule2)"}" ${k==="o"?'stroke="var(--ink2)" stroke-dasharray="4 3"':""}/><text x="${164+w}" y="${y+15}">${fmt(v)}</text>`;});
      sv+="</svg>";
      const am=(r)=>APPS.map((a,i)=>r.app_mpki[i].toFixed(2));
      box.append(h("div",{class:"grid2",style:"margin-top:12px"},
        h("div",{},h("h3",{style:"font-size:15px;margin:0 0 6px"},"Misses on the whole mixture (fewer is better)"),h("div",{html:sv})),
        h("div",{},h("h3",{style:"font-size:15px;margin:0 0 6px"},"MPKI per application"),
          table(["Policy"].concat(APPS),[["LRU"].concat(am(lru)),[bsn.toUpperCase()+" (best simple)"].concat(am(best)),["Selected"].concat(am(sel)),["Belady"].concat(am(bel))],{hl:(r,i)=>i===2}),
          h("p",{class:"note",style:"margin-top:8px"},"Selected: "+polName(sel.name)+". "+spct(sel.vs_best)+" misses vs "+bsn.toUpperCase()+", "+pct(sel.gap,1)+" of the LRU-to-Belady gap closed."))));
    }
  };
  draw();
  return panel(title,badge,note?h("p",{class:"note"},note):null,box);
}
async function renderShared(el){
  el.innerHTML="";
  el.append(h("h2",{},"Four applications sharing the cache"),
    h("p",{class:"lede"},"Hash, graph, matrix and convolution take turns on one last-level cache (10,000 instructions per turn). Models trained on single applications lose here. Mixed-stream DAgger trains them on the merged stream the cache actually sees: mix1 is one round with the oracle's decisions replaced step by step by the learned policy, mix1b first lets LRU and Belady drive the shared cache, mix2 adds a second round."));
  el.append(sharedBlock("val","Validation mixture",BF(),"One mixture of validation inputs. These results were used to pick which version to call the headline version (mix1b) before any test run."));
  const t=sharedBlock("test","Held-out test mixtures",h("span",{class:"full"},"full run, held out"),"Inputs the models never saw. Each mixture was run once, after all choices were made. Nothing was selected on these numbers. The ranking of versions here differs from validation: treat differences of a few points as noise (one mixture, one seed).");
  t.style.borderLeft="4px solid var(--ink)";el.append(t);
  // animation
  const W=await loadWin("shared_default");
  const p=panel("Watch the shared cache",BS());el.append(p);
  if(!W){p.append(winHint("shared_default"));return;}
  p.append(h("p",{class:"note"},"Sample window of test mixture 1 at 2 MB: "+fmt(W.len)+" accesses starting at access "+fmt(W.start)+" of "+fmt(W.trace_len)+" in the merged stream. Colours show which application owns each line; the faded cells are not touched yet."));
  if(!(S3.anim in W.byLabel))S3.anim=W.policies[W.byLabel["gen: mix1b MLP"]!==undefined?W.byLabel["gen: mix1b MLP"]:0].label;
  const sel=select("Policy",W.policies.map(x=>[x.label,x.label]),S3.anim,v=>{S3.anim=v;g.setPolicy(W.byLabel[v]);});
  const die=h("div",{class:"die"}),cnt=h("div");
  const g=new Grid(die,{onUpdate:()=>{pb.upd();upd();}});const pb=playbar(g);
  const pols=()=>{const set=[S3.anim,"LRU","DRRIP","Belady (oracle)"];return [...new Set(set)].filter(l=>l in W.byLabel);};
  function upd(){const n=g.i,ps=pols();const per=ps.map(()=>APPS.map(()=>[0,0]));
    ps.forEach((l,pi)=>{const o=W.policies[W.byLabel[l]].o;for(let j=0;j<n;j++){const a=W.lineApp[W.acc.line[j]];per[pi][a][(o[j]&1)?0:1]++;}});
    const rows=APPS.map((a,ai)=>[h("span",{},h("i",{class:"sw",style:"background:var(--"+a+");margin-right:6px"}),APPNAME[a])].concat(ps.map((l,pi)=>fmt(per[pi][ai][1]))));
    rows.push(["All"].concat(ps.map((l,pi)=>fmt(per[pi].reduce((s,x)=>s+x[1],0)))));
    cnt.replaceChildren(h("p",{class:"num"},"Access "+fmt(n)+" of "+fmt(W.len)+". Misses so far, by policy:"),table(["Application"].concat(ps),rows,{hl:(r,i)=>i===rows.length-1}));}
  p.append(h("div",{class:"row"},sel),h("div",{class:"split",style:"margin-top:10px;grid-template-columns:minmax(0,1fr)"},h("div",{},die,pb.el),cnt));
  g.set(W,W.byLabel[S3.anim]);
}

/* ---------- RESULTS ---------- */
function renderResults(el){
  el.innerHTML="";
  el.append(h("h2",{},"Results and tables"),h("p",{class:"lede"},"The full tables of every experiment, as exported by the simulator. Numbers are computed on complete traces. Test inputs were used once."));
  const nav=h("p",{class:"note"},"Jump to: ");const secs=[];
  const add=(id,title,badge,...kids)=>{const p=panel(title,badge,...kids);p.id="res-"+id;secs.push([id,title]);el.append(p);return p;};
  add("e2","E2: learned policies vs simple policies, held-out inputs",BF(),h("p",{class:"note"},"Gap closed on the 2 test inputs of each application, 2 MB 16-way. 100% = Belady, 0% = LRU."),csvTable("e2_summary",{bars:true}));
  add("e3","E3: one model per application vs one generic model",BF(),h("p",{class:"note"},"MLP: mean ± standard deviation over 3 training seeds. Specialized did not beat generic."),csvTable("e3",{fill:[0],numFrom:2}));
  add("e4","E4: a model used on the wrong application",BF(),h("p",{class:"note"},"Gap closed by the model of the row on the workload of the column. Negative = worse than LRU."),csvTable("e4",{bars:false}));
  add("e5","E5: other cache sizes (models not retrained)",BF(),csvTable("e5",{fill:[0],numFrom:2}));
  add("e6","E6: what a model costs",BF(),h("p",{class:"note"},"Software time on one laptop core. Hardware would have about 100 ns per decision."),csvTable("e6_models"),
    h("h3",{style:"font-size:14px;margin:14px 0 6px"},"Storage: four specialized models vs one generic"),csvTable("e6_storage"));
  add("e7","E7: switching between applications",BF(),csvTable("e7_misses",{fill:[0,1],pol:[2]}));
  add("e8","E8: four applications sharing the LLC, models trained on single applications",BF(),
    h("h3",{style:"font-size:14px;margin:6px 0"},"2 MB, 16-way"),csvTable("e8_default_s10000"),h("h3",{style:"font-size:14px;margin:14px 0 6px"},"8 MB, 16-way"),csvTable("e8_llc8m_16w_s10000"));
  // extensions
  const L=X.lstm_dagger_test,wls=APPS;
  const ex=add("ext","Extension 1: DAgger for the LSTM",BF(),h("p",{class:"note"},"Mean gap closed over the 8 held-out test traces (2 MB). The LSTM without DAgger lagged the MLP; with one DAgger round it matches it, at about 5 times the storage and 10 times the decision time."),
    table(["Model","Specialized (mean of 4 applications)","Generic"].concat(wls.map(w=>APPNAME[w]+" (specialized)")),Object.keys(L.specialized_mean).map(k=>[k,pct(L.specialized_mean[k]),pct(L.generic_mean[k])].concat(wls.map(w=>L.per_workload[k][w]!=null?pct(L.per_workload[k][w]):"-")))),
    h("p",{class:"note"},"Best simple policy on these traces: DRRIP "+pct(L.simple_mean.drrip)+", BIP "+pct(L.simple_mean.bip)+" (mean gap closed)."));
  const ex2=add("ext2","Extension 2: mixed-stream DAgger, every version",BF(),h("p",{class:"note"},"All learned policies on the shared cache, one mixture per table, sorted by share of the LRU-to-Belady gap closed. Open a table to see it."));
  const names={val:"Validation mixture",test:"Held-out test mixture 1",test2:"Held-out test mixture 2"};
  for(const s of ["val","test","test2"])for(const tag of ["default","llc8m"]){const S=X.sets[s][tag];if(!S)continue;
    const rs=S.runs.filter(r=>r.rule).sort((a,b)=>b.gap-a.gap);
    const d=h("details",{},h("summary",{},names[s]+(s==="val"?"":" (run once)")+", "+(tag==="default"?"2 MB":"8 MB")+", best simple "+S.best_simple.toUpperCase()),
      table(["Policy","Misses","Gap closed","vs best simple"].concat(APPS.map(a=>"MPKI "+a)),rs.map(r=>[polName(r.name),fmt(r.misses),pct(r.gap),spct(r.vs_best)].concat(r.app_mpki.map(x=>x.toFixed(2))))));
    ex2.append(d);}
  nav.append(...secs.flatMap(([id,t],i)=>[h("a",{href:"#res-"+id,onclick:e=>{e.preventDefault();$("#res-"+id).scrollIntoView({behavior:"smooth"});}},id.toUpperCase()),i<secs.length-1?" · ":""]));
  el.insertBefore(nav,el.children[2]);
}

/* ---------- router ---------- */
const SCREENS={overview:A.renderOverview,single:A.renderSingle,switching:renderSwitching,shared:renderShared,inspector:A.renderInspector,results:renderResults};
function show(name){
  if(!SCREENS[name])name="overview";
  document.querySelectorAll(".screen").forEach(s=>s.classList.toggle("on",s.id===name));
  document.querySelectorAll(".rail .tab").forEach(b=>b.toggleAttribute("aria-current",b.dataset.s===name)||b.removeAttribute("aria-current"));
  document.querySelectorAll(".rail .tab").forEach(b=>{if(b.dataset.s===name)b.setAttribute("aria-current","page");else b.removeAttribute("aria-current");});
  const el=$("#"+name);if(name!=="overview"||!el.dataset.done){el.dataset.done=1;if(name==="overview"){el.innerHTML="";}SCREENS[name](el);}
  history.replaceState(null,"","#"+name);window.scrollTo(0,0);
}
document.querySelectorAll(".rail .tab").forEach(b=>b.addEventListener("click",()=>show(b.dataset.s)));
$("#themebtn").addEventListener("click",()=>{const r=document.documentElement,cur=r.getAttribute("data-theme")||(matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");
  r.setAttribute("data-theme",cur==="dark"?"light":"dark");const on=document.querySelector(".screen.on");if(on){on.dataset.done="";show(on.id);}});
window.addEventListener("hashchange",()=>show((location.hash||"#overview").slice(1)));
show((location.hash||"#overview").slice(1));

})();
