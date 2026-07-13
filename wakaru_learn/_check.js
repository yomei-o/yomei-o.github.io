
(function(){
  "use strict";
  var cv=document.getElementById('cv'),
      se=document.getElementById('eta'), ve=document.getElementById('veta'),
      s0=document.getElementById('th0'), v0=document.getElementById('vth0'),
      rd=document.getElementById('rd'), btn=document.getElementById('run');
  // 地形 L(θ)=(θ²−1)²+0.3θ, 勾配 L'(θ)=4θ³−4θ+0.3
  function Lf(t){var u=t*t-1; return u*u+0.3*t;}
  function dLf(t){return 4*t*t*t-4*t+0.3;}
  var TH_LO=-1.9, TH_HI=1.9, L_LO=-0.6, L_HI=2.7;
  var MIN_L=-1.035, MIN_R=0.960;   // 谷底の近似位置
  var theta=1.4, traj=[1.4], step=0, raf=null, lastStep=0;

  function setupHiDPI(c){
    var r=window.devicePixelRatio||1, cssW=c.clientWidth||c.width, cssH=c.height*(cssW/c.width);
    c.width=Math.round(cssW*r); c.height=Math.round(cssH*r);
    var ctx=c.getContext('2d'); ctx.setTransform(r,0,0,r,0,0);
    return {ctx:ctx,w:cssW,h:cssH};
  }
  function draw(){
    var d=setupHiDPI(cv), ctx=d.ctx, w=d.w, h=d.h, padL=44, padB=42, padT=16;
    function X(t){return padL+(t-TH_LO)/(TH_HI-TH_LO)*(w-padL-14);}
    function Y(l){return (h-padB)-(l-L_LO)/(L_HI-L_LO)*(h-padB-padT);}
    ctx.clearRect(0,0,w,h);
    // グリッド（L の水平線）
    ctx.strokeStyle='#efe4cd';ctx.lineWidth=1;ctx.fillStyle='#bda877';ctx.font='9px sans-serif';
    for(var l=0;l<=2;l++){var y=Y(l);ctx.beginPath();ctx.moveTo(padL,y);ctx.lineTo(w-8,y);ctx.stroke();ctx.fillText('L='+l,6,y-3);}
    var y0=Y(0);ctx.beginPath();ctx.moveTo(padL,y0);ctx.lineTo(w-8,y0);ctx.stroke();
    // 地形曲線
    ctx.strokeStyle='#b06a10';ctx.lineWidth=2.6;ctx.beginPath();var f=true;
    for(var i=0;i<=240;i++){var t=TH_LO+(TH_HI-TH_LO)*i/240; var x=X(t),y=Y(Lf(t)); if(f){ctx.moveTo(x,y);f=false;}else ctx.lineTo(x,y);}
    ctx.stroke();
    // 谷底マーカー
    function well(t,label,color){var x=X(t),y=Y(Lf(t));
      ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,3.5,0,7);ctx.fill();
      ctx.fillStyle='#8a7a5a';ctx.font='11px sans-serif';ctx.fillText(label,x-18,y+18);}
    well(MIN_L,'深い谷','#3d7a3a'); well(MIN_R,'浅い谷','#a86518');
    // 軸ラベル
    ctx.fillStyle='#6b5a38';ctx.font='12px sans-serif';
    ctx.fillText('θ（パラメータ）→', w-140, h-padB+26);
    ctx.save();ctx.translate(14,h/2+30);ctx.rotate(-Math.PI/2);ctx.fillText('L（損失＝高さ）',0,0);ctx.restore();
    // 軌跡（薄い点）
    for(var k=0;k<traj.length;k++){var tt=traj[k];
      ctx.fillStyle='rgba(176,106,16,'+(0.12+0.25*k/Math.max(1,traj.length))+')';
      ctx.beginPath();ctx.arc(X(tt),Y(Lf(tt)),2.4,0,7);ctx.fill();}
    // 降下方向の矢印 −∇L（水平成分で表示）
    var g=dLf(theta), dir=(g>0?-1:1); // −∇L の符号
    var bx=X(theta), by=Y(Lf(theta));
    if(isFinite(bx)&&Math.abs(theta)<=TH_HI){
      var aLen=Math.min(46, 10+Math.abs(g)*7);
      ctx.strokeStyle='#3d7a3a';ctx.lineWidth=2.2;ctx.beginPath();
      ctx.moveTo(bx,by);ctx.lineTo(bx+dir*aLen,by);ctx.stroke();
      ctx.beginPath();ctx.moveTo(bx+dir*aLen,by);ctx.lineTo(bx+dir*(aLen-7),by-4);
      ctx.lineTo(bx+dir*(aLen-7),by+4);ctx.closePath();ctx.fillStyle='#3d7a3a';ctx.fill();
    }
    // ボール
    if(isFinite(bx)&&Math.abs(theta)<=TH_HI){
      ctx.fillStyle='#c0392b';ctx.beginPath();ctx.arc(bx,by,6.5,0,7);ctx.fill();
      ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
    }
  }
  function report(status){
    var eta=parseFloat(se.value);
    if(status==='diverge'){
      rd.className='readout bad';
      rd.textContent='発散：η='+eta.toFixed(3)+' が大きすぎて、ボールが坂の外へ飛び出した（|1−ηk|>1）。歩幅を小さくしてみてください。';
    } else if(status==='slow'){
      rd.className='readout local';
      if(lastStep>0.05){
        rd.textContent='収束せず振動：η='+eta.toFixed(3)+' がこの谷の急さに対して大きすぎ、谷底を跨いで行き来しています（現在 θ='+theta.toFixed(3)+'、1歩の幅≈'+lastStep.toFixed(3)+'）。歩幅を下げてみてください。';
      } else {
        rd.textContent='まだ谷に着いていない：η='+eta.toFixed(3)+' が小さすぎて、500ステップでも収束しませんでした（現在 θ='+theta.toFixed(3)+'）。歩幅を上げてみてください。';
      }
    } else {
      var Lv=Lf(theta);
      if(theta<-0.3){
        rd.className='readout ok';
        rd.textContent='深い谷（大域的な最小）に到達：θ='+theta.toFixed(3)+'、L='+Lv.toFixed(3)+'。'+step+' ステップで収束。これが一番低い谷 ── 大当たり。';
      } else if(theta>0.3){
        rd.className='readout local';
        rd.textContent='浅い谷（局所的な最小）で停止：θ='+theta.toFixed(3)+'、L='+Lv.toFixed(3)+'。'+step+' ステップで収束。∇L=0 だがここは最良ではない ── 隣にもっと深い谷がある。開始位置を左へずらすと届きます。';
      } else {
        rd.className='readout local';
        rd.textContent='尾根の近くで停止：θ='+theta.toFixed(3)+'。勾配がほぼ0の不安定な点。開始位置を少しずらしてみてください。';
      }
    }
  }
  function frame(){
    var eta=parseFloat(se.value);
    var g=dLf(theta);
    lastStep=Math.abs(eta*g);
    theta = theta - eta*g;
    step++;
    if(!isFinite(theta)||Math.abs(theta)>3){ theta=(theta>0?1:-1)*3; draw(); report('diverge'); return; }
    traj.push(theta);
    if(traj.length>600) traj.shift();
    draw();
    if(lastStep<1e-4){ report('converge'); return; }
    if(step>=500){ report('slow'); return; }
    raf=requestAnimationFrame(frame);
  }
  function start(){
    if(raf) cancelAnimationFrame(raf);
    theta=parseFloat(s0.value); traj=[theta]; step=0;
    rd.className='readout local'; rd.textContent='学習中… ボールが足もとの傾き −∇L にしたがって坂を下ります。';
    raf=requestAnimationFrame(frame);
  }
  function syncLabels(){ ve.textContent=parseFloat(se.value).toFixed(3); v0.textContent=parseFloat(s0.value).toFixed(2); }
  se.addEventListener('input',function(){syncLabels();});
  s0.addEventListener('input',function(){syncLabels(); if(raf)cancelAnimationFrame(raf); theta=parseFloat(s0.value); traj=[theta]; step=0; draw();
    rd.className='readout local'; rd.textContent='開始位置を θ='+theta.toFixed(2)+' に設定。「学習スタート」を押すと下り始めます。';});
  btn.addEventListener('click',start);
  window.addEventListener('resize',draw);
  syncLabels();
  if(window.MathJax&&MathJax.startup&&MathJax.startup.promise){MathJax.startup.promise.then(function(){theta=parseFloat(s0.value);traj=[theta];draw();});}
  setTimeout(function(){theta=parseFloat(s0.value);traj=[theta];draw();},60);
  window.addEventListener('load',function(){theta=parseFloat(s0.value);traj=[theta];draw();});
})();
