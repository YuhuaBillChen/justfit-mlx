const slots={A:[0,2,5,9,12,16],B:[1,7],C:[3,6,10,14],D:[4],E:[1,7,8,11,13,15]};
const frames=[
 {active:["A"],detail:"A begins alone and owns six equal-capacity page groups.",mode:"B1 · MTP",head:"generation lease",drafter:"attached",vision:"detached",note:"A's six page IDs remain fixed across serving-mode changes."},
 {active:["A","B","C","D"],detail:"Unequal requests share thirteen groups. E needs six, but only five are free and B4 is full.",mode:"B4 · AR",head:"generation leases",drafter:"detached",vision:"detached",note:"E waits for both a lane and enough admission capacity."},
 {active:["A","C","D","E"],detail:"B completes. After evaluation and reference release, E reuses B's two old slots plus four free slots.",mode:"B4 · AR",head:"generation leases",drafter:"detached",vision:"detached",note:"Outlined slots were B's and are now owned by E."},
 {active:["A","E"],detail:"Tiny D and medium C leave. Their slots return to the pool free list; A and E do not move.",mode:"B2 · AR",head:"generation leases",drafter:"detached",vision:"detached",note:"Logical page release makes fixed pool slots reusable; pool backing remains resident."},
 {active:["A"],detail:"E completes. A is singleton again, so the drafter can return after the safe cohort boundary.",mode:"B1 · MTP resumes",head:"generation lease",drafter:"attached",vision:"detached",note:"A resumes MTP with the same target KV and recurrent state."},
 {active:[],detail:"A completes and the text cohort closes. Image request F can enter its separate media-embedding phase.",mode:"Vision phase",head:"detached",drafter:"detached",vision:"attached",note:"No shown KV owner does not mean zero memory use: vision is active while the resident pool remains allocated."}
];

// `lever` names which segment of the budget bar each frame is exercising, so
// the flat bar lights up in step with the 3D scene: the mechanism doing work
// right now is the one keeping that slice of memory unmaterialized.
const levers = ["kv", "kv", "kv", "kv", "comp", "comp"];

let pool3d = null; // set once the 3D module loads; render() tolerates null

function render(index) {
  const frame = frames[index];
  const owners = Array(18).fill("");
  frame.active.forEach((id) => slots[id].forEach((slot) => (owners[slot] = id)));
  const attached = [
    frame.head !== "detached",
    frame.drafter === "attached",
    frame.vision === "attached",
  ];
  if (pool3d) pool3d.setFrame(owners, attached);

  const used = owners.filter(Boolean).length;
  document.querySelector("#pool-count").textContent = `${used} / 18 pages`;
  document.querySelector("#stage-caption").textContent = frame.detail;
  document.querySelector("#run-mode").textContent = frame.mode;

  document.querySelectorAll(".chip[data-owner]").forEach((chip) => {
    chip.classList.toggle("is-on", frame.active.includes(chip.dataset.owner));
  });
  document.querySelectorAll(".chip[data-comp]").forEach((chip) => {
    chip.classList.toggle("is-on", attached[Number(chip.dataset.comp)]);
  });
  document.querySelectorAll(".seg[data-lever]").forEach((seg) => {
    seg.classList.toggle("is-lit", seg.dataset.lever === levers[index]);
  });
  document.querySelectorAll("#event-tabs button").forEach((button, i) => {
    button.setAttribute("aria-pressed", String(i === index));
  });
}

let currentIndex = 0;
let auto = null;

function show(index, manual) {
  currentIndex = (index + frames.length) % frames.length;
  render(currentIndex);
  if (manual && auto) {
    clearInterval(auto); // a click takes over; stop advancing on its own
    auto = null;
  }
}

document.querySelectorAll("#event-tabs button").forEach((button) =>
  button.addEventListener("click", () => show(Number(button.dataset.frame), true))
);

render(currentIndex);
if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  auto = setInterval(() => show(currentIndex + 1), 2600);
}

window.addEventListener("load", () => {
  import("./pool3d.js").then((m) => {
    pool3d = m.mountPagePool(document.getElementById("pool"));
    render(currentIndex);
  });
});

// Install-block tab switcher (Quick Start / Full pinned reproduce).
document.querySelectorAll(".install-tabs button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".install-tabs button").forEach((b) => b.classList.toggle("active", b === button));
    const tab = button.dataset.tab;
    document.querySelectorAll(".install-block").forEach((pre) => {
      pre.hidden = pre.dataset.panel !== tab;
    });
  });
});

// Copy the currently visible install block to the clipboard.
const copyBtn = document.querySelector(".copy-btn");
if (copyBtn) {
  copyBtn.addEventListener("click", async () => {
    const visible = document.querySelector(".install-block:not([hidden])");
    if (!visible) return;
    try {
      await navigator.clipboard.writeText(visible.textContent.trim());
      const original = copyBtn.textContent;
      copyBtn.textContent = "Copied";
      setTimeout(() => (copyBtn.textContent = original), 1400);
    } catch {
      /* clipboard API unavailable (e.g. non-secure context) — silent no-op */
    }
  });
}
