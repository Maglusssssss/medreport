/* app.js — MedReport AI Frontend Logic */

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile = null;
let lastReport   = "";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const uploadZone  = document.getElementById("uploadZone");
const fileInput   = document.getElementById("fileInput");
const placeholder = document.getElementById("placeholder");
const previewBox  = document.getElementById("previewBox");
const previewImg  = document.getElementById("previewImg");
const btnRemove   = document.getElementById("btnRemove");
const btnSubmit   = document.getElementById("btnSubmit");
const modelTag    = document.getElementById("modelTag");

// ── Kiểm tra model ngay khi load ─────────────────────────────────────────────
fetch("/health")
  .then(r => r.json())
  .then(d => {
    if (d.model_ready) {
      modelTag.textContent = "✓ Model sẵn sàng";
      modelTag.className   = "htag tag-ok";
    } else {
      modelTag.textContent = "⚠ Chưa train";
      modelTag.className   = "htag tag-warn";
      btnSubmit.disabled   = true;
    }
  })
  .catch(() => {
    modelTag.textContent = "⚠ Lỗi kết nối";
    modelTag.className   = "htag tag-err";
  });

// ── Upload handlers ───────────────────────────────────────────────────────────
uploadZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", e => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
});

uploadZone.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

btnRemove.addEventListener("click", e => { e.stopPropagation(); resetForm(); });

function loadFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Vui lòng chọn file ảnh (PNG, JPG, JPEG).");
    return;
  }
  selectedFile = file;
  const url    = URL.createObjectURL(file);
  previewImg.src = url;
  placeholder.style.display = "none";
  previewBox.style.display  = "block";
  btnSubmit.disabled = false;
}

// ── Submit ────────────────────────────────────────────────────────────────────
btnSubmit.addEventListener("click", async () => {
  if (!selectedFile) { alert("Vui lòng chọn ảnh X-Quang."); return; }

  setUIState("loading");
  animateLoadingSteps();

  const form = new FormData();
  form.append("image", selectedFile);

  try {
    const res  = await fetch("/predict", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok || !data.success) throw new Error(data.error || "Lỗi server");

    renderResult(data);
    setUIState("result");

  } catch (err) {
    document.getElementById("errorMsg").textContent = err.message;
    setUIState("error");
  }
});

// ── Render kết quả ────────────────────────────────────────────────────────────
function renderResult(data) {
  lastReport = data.report || "";

  // Thumbnail
  const thumb = document.getElementById("thumbImg");
  if (data.thumb) thumb.src = data.thumb;

  // Meta info
  let meta = `<strong>Kích thước:</strong> ${data.img_size || "—"}`;
  document.getElementById("metaInfo").innerHTML = meta;

  // Report meta
  document.getElementById("resultMeta").textContent =
    new Date().toLocaleString("vi-VN");

  // Báo cáo
  document.getElementById("reportText").textContent = lastReport;

  // Từ khoá bệnh lý
  const tagsEl   = document.getElementById("findingsTags");
  const findings = data.findings || [];
  if (findings.length === 0) {
    tagsEl.innerHTML = `<span class="ftag ftag-normal">Không phát hiện bất thường rõ ràng</span>`;
  } else {
    tagsEl.innerHTML = findings.map(f => {
      const cls = f.level === "severe" ? "ftag-severe"
                : f.level === "mild"   ? "ftag-mild"
                : "ftag-normal";
      return `<span class="ftag ${cls}">${f.label}</span>`;
    }).join("");
  }

  // Khuyến nghị dựa trên bệnh lý tìm được
  const recEl = document.getElementById("recommendBox");
  const recs  = buildRecommendations(findings);
  recEl.innerHTML = recs.map(r =>
    `<div class="rec-item"><span>${r.icon}</span><span>${r.text}</span></div>`
  ).join("");
}

function buildRecommendations(findings, symptoms) {
  const recs    = [];
  const levels  = findings.map(f => f.level);

  if (levels.includes("severe")) {
    recs.push({ icon: "🚨", text: "Phát hiện dấu hiệu nghiêm trọng — cần hội chẩn bác sĩ ngay." });
    recs.push({ icon: "🏥", text: "Cân nhắc chuyển viện hoặc nhập viện để theo dõi và điều trị." });
  } else if (levels.includes("mild")) {
    recs.push({ icon: "👨‍⚕️", text: "Nên gặp bác sĩ chuyên khoa hô hấp trong 1–2 tuần tới." });
    recs.push({ icon: "📅", text: "Chụp phim kiểm tra lại sau 4–6 tuần điều trị." });
  } else {
    recs.push({ icon: "✅", text: "Hình ảnh không có bất thường rõ ràng — duy trì tầm soát định kỳ." });
  }

  if (symptoms && symptoms.toLowerCase().includes("ho")) {
    recs.push({ icon: "💊", text: "Cân nhắc xét nghiệm đờm và chức năng hô hấp nếu ho kéo dài > 3 tuần." });
  }

  recs.push({ icon: "ℹ️", text: "Kết quả AI chỉ mang tính hỗ trợ. Chẩn đoán cuối cùng do bác sĩ quyết định." });
  return recs;
}

// ── UI States ─────────────────────────────────────────────────────────────────
function setUIState(state) {
  document.getElementById("stateEmpty").style.display   = "none";
  document.getElementById("stateLoading").style.display = "none";
  document.getElementById("stateError").style.display   = "none";
  document.getElementById("stateResult").style.display  = "none";

  if (state === "empty") {
    document.getElementById("stateEmpty").style.display = "flex";
  } else if (state === "loading") {
    document.getElementById("stateLoading").style.display = "flex";
    btnSubmit.disabled = true;
    btnSubmit.textContent = "Đang phân tích...";
  } else if (state === "error") {
    document.getElementById("stateError").style.display = "flex";
    btnSubmit.disabled = false;
    btnSubmit.textContent = "⚡ Sinh báo cáo chẩn đoán";
  } else if (state === "result") {
    document.getElementById("stateResult").style.display = "block";
    btnSubmit.disabled = false;
    btnSubmit.textContent = "⚡ Sinh báo cáo chẩn đoán";
    //loadMetrics();
  }
}

// ── Loading steps animation ───────────────────────────────────────────────────
function animateLoadingSteps() {
  const steps = ["ls1", "ls2", "ls3", "ls4"];
  steps.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.className = "lstep";
  });

  let i = 0;
  const interval = setInterval(() => {
    if (i > 0) {
      const prev = document.getElementById(steps[i - 1]);
      if (prev) prev.className = "lstep done";
    }
    if (i < steps.length) {
      const cur = document.getElementById(steps[i]);
      if (cur) cur.className = "lstep active";
      i++;
    } else {
      clearInterval(interval);
    }
  }, 600);
}

// ── Load metrics từ file kết quả (nếu có) ────────────────────────────────────
/*
function loadMetrics() {
  fetch("/static/eval_results.json")
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      if (!data || !data.metrics) return;
      const card = document.getElementById("metricsCard");
      const grid = document.getElementById("metricsGrid");
      const m    = data.metrics;

      grid.innerHTML = Object.entries(m).map(([k, v]) =>
        `<div class="metric-item">
           <div class="metric-val">${(v * 100).toFixed(1)}</div>
           <div class="metric-key">${k}</div>
         </div>`
      ).join("");

      card.style.display = "block";
    })
    .catch(() => {});
} */

// ── Actions ───────────────────────────────────────────────────────────────────
function copyReport() {
  if (!lastReport) return;
  navigator.clipboard.writeText(lastReport).then(() => {
    const btn = event.target;
    btn.textContent = "✓ Đã sao chép!";
    setTimeout(() => { btn.textContent = "Sao chép báo cáo"; }, 2000);
  });
}

function resetForm() {
  selectedFile      = null;
  lastReport        = "";
  fileInput.value   = "";
  previewImg.src    = "";
  previewBox.style.display  = "none";
  placeholder.style.display = "flex";
  document.getElementById("age").value      = "";
  document.getElementById("gender").value   = "";
  document.getElementById("symptoms").value = "";
  // document.getElementById("metricsCard").style.display = "none";
  setUIState("empty");
  btnSubmit.disabled    = true;
  btnSubmit.textContent = "⚡ Sinh báo cáo chẩn đoán";
}
