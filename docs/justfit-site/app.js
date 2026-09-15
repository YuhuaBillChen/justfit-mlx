const slots={A:[0,2,5,9,12,16],B:[1,7],C:[3,6,10,14],D:[4],E:[1,7,8,11,13,15]};
const frames=[
 {active:["A"],detail:"A begins alone and owns six equal-capacity page groups.",mode:"B1 · MTP",head:"generation lease",drafter:"attached",vision:"detached",note:"A's six page IDs remain fixed across serving-mode changes."},
 {active:["A","B","C","D"],detail:"Unequal requests share thirteen groups. E needs six, but only five are free and B4 is full.",mode:"B4 · AR",head:"generation leases",drafter:"detached",vision:"detached",note:"E waits for both a lane and enough admission capacity."},
 {active:["A","C","D","E"],detail:"B completes. After evaluation and reference release, E reuses B's two old slots plus four free slots.",mode:"B4 · AR",head:"generation leases",drafter:"detached",vision:"detached",note:"Outlined slots were B's and are now owned by E."},
 {active:["A","E"],detail:"Tiny D and medium C leave. Their slots return to the pool free list; A and E do not move.",mode:"B2 · AR",head:"generation leases",drafter:"detached",vision:"detached",note:"Logical page release makes fixed pool slots reusable; pool backing remains resident."},
 {active:["A"],detail:"E completes. A is singleton again, so the drafter can return after the safe cohort boundary.",mode:"B1 · MTP resumes",head:"generation lease",drafter:"attached",vision:"detached",note:"A resumes MTP with the same target KV and recurrent state."},
 {active:[],detail:"A completes and the text cohort closes. Image request F can enter its separate media-embedding phase.",mode:"Vision phase",head:"detached",drafter:"detached",vision:"attached",note:"No shown KV owner does not mean zero memory use: vision is active while the resident pool remains allocated."}
];
const pool=document.querySelector("#pool");
function render(index){
 const frame=frames[index],owners=Array(18).fill("");
 frame.active.forEach(id=>slots[id].forEach(slot=>owners[slot]=id));
 pool.replaceChildren(...owners.map((owner,i)=>{
   const el=document.createElement("div");el.className="slot";el.dataset.owner=owner||"free";
   if(index===2&&slots.B.includes(i))el.dataset.reused="true";
   el.innerHTML=`<small>${String(i+1).padStart(2,"0")}</small><b>${owner||"—"}</b>`;return el;
 }));
 const used=owners.filter(Boolean).length;
 document.querySelector("#pool-count").textContent=`${used} occupied / ${18-used} free`;
 document.querySelector("#event-detail").textContent=frame.detail;
 document.querySelector("#pool-note").textContent=frame.note;
 for(const key of ["mode","head","drafter","vision"])document.querySelector(`#${key}`).textContent=frame[key];
 document.querySelectorAll("#event-tabs button").forEach((button,i)=>button.setAttribute("aria-pressed",String(i===index)));
}
document.querySelectorAll("#event-tabs button").forEach(button=>button.addEventListener("click",()=>render(Number(button.dataset.frame))));
render(0);
