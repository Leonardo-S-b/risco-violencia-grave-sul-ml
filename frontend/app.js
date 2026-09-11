const form = document.querySelector("#consulta-form");
const ufSelect = document.querySelector("#uf");
const citySelect = document.querySelector("#municipio");
const periodSelect = document.querySelector("#periodo");
const submitButton = document.querySelector("#consultar");
const buttonLabel = submitButton.querySelector(".button-label");
const formError = document.querySelector("#form-error");

const states = {
  empty: document.querySelector("#empty-state"),
  loading: document.querySelector("#loading-state"),
  error: document.querySelector("#error-state"),
  success: document.querySelector("#success-state"),
};

const clientCache = {
  municipalities: new Map(),
  periods: new Map(),
};

let retryAction = null;

function setView(activeState) {
  Object.entries(states).forEach(([name, element]) => {
    element.hidden = name !== activeState;
  });
}

function titleCaseCity(value) {
  const lowercaseWords = new Set(["DA", "DAS", "DE", "DO", "DOS", "E"]);
  return value
    .toLocaleLowerCase("pt-BR")
    .split(" ")
    .map((word, index) => {
      if (index > 0 && lowercaseWords.has(word.toLocaleUpperCase("pt-BR"))) return word;
      return word.charAt(0).toLocaleUpperCase("pt-BR") + word.slice(1);
    })
    .join(" ");
}

function formatMonth(value) {
  const [year, month] = value.split("-").map(Number);
  const formatted = new Intl.DateTimeFormat("pt-BR", {
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(Date.UTC(year, month - 1, 1)));
  return formatted.charAt(0).toLocaleUpperCase("pt-BR") + formatted.slice(1);
}

function formatMonthShort(value) {
  const [year, month] = value.split("-").map(Number);
  return new Intl.DateTimeFormat("pt-BR", {
    month: "short",
    year: "2-digit",
    timeZone: "UTC",
  }).format(new Date(Date.UTC(year, month - 1, 1)));
}

function formatNumber(value, maximumFractionDigits = 0) {
  return Number(value).toLocaleString("pt-BR", { maximumFractionDigits });
}

async function apiFetch(url, options) {
  const response = await fetch(url, options);
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const error = new Error(payload?.detail || "O servidor não conseguiu concluir a solicitação.");
    error.status = response.status;
    error.kind = "api";
    throw error;
  }
  return { payload, response };
}

function setSelectLoading(select, message) {
  select.disabled = true;
  select.replaceChildren(new Option(message, ""));
}

function showFormError(message) {
  formError.textContent = message;
  formError.hidden = false;
}

function clearFormError() {
  formError.hidden = true;
  formError.textContent = "";
  [ufSelect, citySelect, periodSelect].forEach((field) => field.removeAttribute("aria-invalid"));
}

function updateSubmitState() {
  submitButton.disabled = !citySelect.value || !periodSelect.value;
}

async function loadMunicipalities(preferredCity = "CURITIBA") {
  clearFormError();
  const uf = ufSelect.value;
  setSelectLoading(citySelect, "Carregando municípios...");
  setSelectLoading(periodSelect, "Selecione primeiro o município");
  updateSubmitState();

  try {
    let municipalities = clientCache.municipalities.get(uf);
    if (!municipalities) {
      const { payload } = await apiFetch(`/api/v1/municipios?uf=${encodeURIComponent(uf)}`);
      municipalities = payload.municipios;
      clientCache.municipalities.set(uf, municipalities);
    }
    citySelect.replaceChildren(
      ...municipalities.map((city) => new Option(titleCaseCity(city), city))
    );
    citySelect.disabled = false;
    if (municipalities.includes(preferredCity)) citySelect.value = preferredCity;
    await loadPeriods();
  } catch (error) {
    setSelectLoading(citySelect, "Não foi possível carregar");
    showFormError(error.message);
  }
}

async function loadPeriods() {
  clearFormError();
  const uf = ufSelect.value;
  const city = citySelect.value;
  if (!city) return;

  setSelectLoading(periodSelect, "Carregando períodos...");
  updateSubmitState();
  const key = `${uf}|${city}`;

  try {
    let periods = clientCache.periods.get(key);
    if (!periods) {
      const query = new URLSearchParams({ uf, municipio: city });
      const { payload } = await apiFetch(`/api/v1/periodos?${query}`);
      periods = payload.periodos;
      clientCache.periods.set(key, periods);
    }
    periodSelect.replaceChildren(
      ...periods.slice().reverse().map((period) => new Option(formatMonth(period), period))
    );
    periodSelect.disabled = false;
    updateSubmitState();
  } catch (error) {
    setSelectLoading(periodSelect, "Não foi possível carregar");
    showFormError(error.message);
  }
}

function renderResult(result, cameFromCache) {
  const hasAlert = result.alerta;
  const probability = Math.max(0, Math.min(1, result.probabilidade));
  const threshold = Math.max(0, Math.min(1, result.limiar));
  const statusBadge = document.querySelector("#status-badge");
  const cacheBadge = document.querySelector("#cache-badge");

  statusBadge.className = `status-badge ${hasAlert ? "alert" : "safe"}`;
  statusBadge.textContent = hasAlert ? "Com alerta" : "Sem alerta";
  cacheBadge.hidden = !cameFromCache;

  document.querySelector("#result-classification").textContent = hasAlert
    ? "Este município pede uma análise complementar."
    : "Nenhum alerta foi identificado neste período.";
  document.querySelector("#result-message").textContent = result.mensagem;
  document.querySelector("#probability-value").textContent = probability.toLocaleString("pt-BR", {
    style: "percent",
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
  document.querySelector("#probability-bar").style.width = `${probability * 100}%`;
  document.querySelector("#threshold-mark").style.left = `${threshold * 100}%`;
  document.querySelector("#threshold-label").textContent = `Limiar ${(threshold * 100).toLocaleString("pt-BR", { maximumFractionDigits: 2 })}%`;
  document.querySelector(".meter").setAttribute("aria-valuenow", String(Math.round(probability * 100)));
  document.querySelector("#detail-city").textContent = `${titleCaseCity(result.municipio)} - ${result.uf}`;
  document.querySelector("#detail-period").textContent = formatMonth(result.data_referencia);
  document.querySelector("#result-subtitle").textContent = `${titleCaseCity(result.municipio)} - ${result.uf}`;
  if (result.contexto && document.querySelector("#context-population")) {
    renderContext(result.contexto, result.uf);
  }
  setView("success");
}

function renderContext(context, uf) {
  document.querySelector("#context-population").textContent = formatNumber(context.populacao);
  document.querySelector("#context-current-total").textContent = formatNumber(context.total_mes_referencia);
  document.querySelector("#context-rate-6m").textContent = formatNumber(context.taxa_ultimos_6m_100k, 2);
  document.querySelector("#context-rate-12m").textContent = formatNumber(context.taxa_ultimos_12m_100k, 2);

  const trendBadge = document.querySelector("#trend-badge");
  const trendMeta = {
    aumento: { label: "Aumento", className: "up", symbol: "↑" },
    queda: { label: "Queda", className: "down", symbol: "↓" },
    estabilidade: { label: "Estabilidade", className: "", symbol: "→" },
  }[context.tendencia_6m] || { label: "Sem tendência", className: "", symbol: "—" };
  const variation = context.variacao_6m_percentual;
  trendBadge.className = `trend-badge ${trendMeta.className}`.trim();
  trendBadge.textContent = variation === null
    ? `${trendMeta.symbol} ${trendMeta.label}`
    : `${trendMeta.symbol} ${trendMeta.label} de ${formatNumber(Math.abs(variation), 1)}%`;
  trendBadge.title = "Comparação dos 6 meses recentes com os 6 meses anteriores";

  const stateRate = context.taxa_estado_ultimos_12m_100k;
  const ratio = context.razao_municipio_estado;
  document.querySelector("#state-comparison-value").textContent = ratio === null
    ? "Comparação indisponível"
    : `${formatNumber(ratio, 2)}× a taxa de ${uf}`;
  document.querySelector("#state-comparison-detail").textContent =
    `Taxa estadual: ${formatNumber(stateRate, 2)} por 100 mil habitantes.`;

  renderHistory(context.historico_12m);
  renderComposition(context.composicao_12m);
  document.querySelector("#principal-category").textContent = context.principal_categoria_12m;
}

function renderHistory(history) {
  const chart = document.querySelector("#history-chart");
  const maximum = Math.max(...history.map((point) => point.total), 1);
  chart.replaceChildren(
    ...history.map((point) => {
      const bar = document.createElement("span");
      bar.className = "history-bar";
      bar.style.height = `${Math.max((point.total / maximum) * 100, 4)}%`;
      bar.setAttribute("role", "img");
      bar.setAttribute(
        "aria-label",
        `${formatMonth(point.periodo)}: ${formatNumber(point.total)} registros`
      );
      bar.title = `${formatMonth(point.periodo)}: ${formatNumber(point.total)}`;
      return bar;
    })
  );
  const total = history.reduce((sum, point) => sum + point.total, 0);
  document.querySelector("#history-total").textContent = `${formatNumber(total)} em 12 meses`;
  document.querySelector("#history-start").textContent = formatMonthShort(history[0].periodo);
  document.querySelector("#history-end").textContent = formatMonthShort(history.at(-1).periodo);
}

function renderComposition(composition) {
  const list = document.querySelector("#composition-list");
  const maximum = Math.max(...composition.map((item) => item.total_12m), 1);
  list.replaceChildren(
    ...composition.map((item) => {
      const row = document.createElement("li");
      const meta = document.createElement("div");
      const label = document.createElement("span");
      const value = document.createElement("strong");
      const track = document.createElement("div");
      const fill = document.createElement("span");

      meta.className = "composition-meta";
      track.className = "composition-track";
      fill.className = "composition-fill";
      label.textContent = item.categoria;
      value.textContent = formatNumber(item.total_12m);
      fill.style.width = `${(item.total_12m / maximum) * 100}%`;
      meta.append(label, value);
      track.append(fill);
      row.append(meta, track);
      return row;
    })
  );
}

function renderError(error, action) {
  retryAction = action;
  document.querySelector("#error-message").textContent =
    error instanceof TypeError
      ? "A API parece estar indisponível. Verifique se o servidor está em execução."
      : error.message;
  setView("error");
}

async function submitPrediction() {
  clearFormError();
  if (!citySelect.value || !periodSelect.value) {
    showFormError("Selecione o município e o mês de referência.");
    if (!citySelect.value) citySelect.setAttribute("aria-invalid", "true");
    if (!periodSelect.value) periodSelect.setAttribute("aria-invalid", "true");
    return;
  }

  setView("loading");
  submitButton.disabled = true;
  buttonLabel.textContent = "Analisando dados...";
  const body = {
    uf: ufSelect.value,
    municipio: citySelect.value,
    data_referencia: periodSelect.value,
  };

  let apiResult;
  try {
    apiResult = await apiFetch("/api/v1/predicoes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    renderError(error, submitPrediction);
    return;
  }

  try {
    renderResult(
      apiResult.payload,
      apiResult.response.headers.get("X-Cache") === "HIT"
    );
  } catch (error) {
    console.error("Falha ao exibir o resultado recebido:", error);
    renderError(
      new Error("O resultado chegou, mas a página está desatualizada. Atualize com Ctrl + F5."),
      () => window.location.reload()
    );
  } finally {
    buttonLabel.textContent = "Analisar município";
    updateSubmitState();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitPrediction();
});

ufSelect.addEventListener("change", () => loadMunicipalities(""));
citySelect.addEventListener("change", loadPeriods);
periodSelect.addEventListener("change", updateSubmitState);
document.querySelector("#retry-button").addEventListener("click", () => retryAction?.());

loadMunicipalities();
