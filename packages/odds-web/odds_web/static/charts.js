// Draws the candidate page's charts from the series each `.chart` element carries.
(() => {
  const colour = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const frame = (element, extra) => ({
    width: element.clientWidth,
    marginLeft: 48,
    marginRight: 16,
    style: {
      background: "transparent",
      fontFamily: colour("--font-mono"),
      fontSize: "10px",
      color: colour("--muted-foreground"),
    },
    ...extra,
  });

  const evChart = (element, data) => {
    const muted = colour("--muted-foreground");
    const line = colour("--chart-1");
    const checkpointed = new Set(data.intervals.filter((d) => d.checkpoint).map((d) => d.n));
    // A current estimate at a checkpoint's n but another family size sits just left of it, inside
    // the frame even at the last checkpoint.
    const besideCheckpoint = (d) => !d.checkpoint && checkpointed.has(d.n);
    const intervals = (rows, dx) => [
      Plot.ruleX(rows.filter((d) => d.promote), { x: "n", dx, y1: (d) => d.promote[0], y2: (d) => d.promote[1], stroke: colour("--foreground"), strokeOpacity: 0.25, strokeWidth: 6, strokeLinecap: "round", clip: true }),
      Plot.ruleX(rows, { x: "n", dx, y1: (d) => d.ci[0], y2: (d) => d.ci[1], stroke: colour("--foreground"), strokeWidth: 2, clip: true }),
      Plot.dot(rows, {
        x: "n",
        y: "mean",
        dx,
        fill: line,
        stroke: colour("--background"),
        strokeWidth: 2,
        r: 4,
        clip: true,
        title: (d) => `${d.label ?? (d.checkpoint ? "checkpoint" : "current")} n=${d.n}\nmean EV ${d.mean.toFixed(1)}%\n${Math.round(d.level * 100)}% CI ${d.ci[0].toFixed(1)} … ${d.ci[1].toFixed(1)}` + (d.promote ? `\npromote CI ${d.promote[0].toFixed(1)} … ${d.promote[1].toFixed(1)}` : ""),
      }),
    ];
    const reach = Math.max(...data.checkpoints, data.running.length);
    return Plot.plot(
      frame(element, {
        height: 190,
        x: { domain: [0, reach], label: "bets with a close", grid: false },
        y: { domain: [-data.axis, data.axis], label: null, tickFormat: (v) => `${v > 0 ? "+" : ""}${v}%` },
        marks: [
          Plot.ruleY([0], { stroke: muted, strokeDasharray: "2 3" }),
          Plot.ruleX(data.checkpoints, { stroke: muted, strokeDasharray: "4 3" }),
          Plot.text(data.checkpoints, { x: (n) => n, y: data.axis, text: (n) => `${n}`, dy: 8, textAnchor: "end", dx: -3 }),
          Plot.line(data.running, { x: (d) => d[0], y: (d) => d[1], stroke: line, strokeWidth: 2, clip: true }),
          ...(data.estimate ? [Plot.line(data.estimate, { x: (d) => d[0], y: (d) => d[1], stroke: colour("--chart-2"), strokeWidth: 2, strokeDasharray: "5 3", clip: true })] : []),
          ...intervals(data.intervals.filter((d) => !besideCheckpoint(d)), 0),
          ...intervals(data.intervals.filter(besideCheckpoint), -10),
        ],
      }),
    );
  };

  const cumulativeChart = (element, data) => {
    const muted = colour("--muted-foreground");
    return Plot.plot(
      frame(element, {
        height: 170,
        x: { label: "bets with a close", grid: false },
        y: { label: null, tickFormat: (v) => `${v > 0 ? "+" : ""}${v}u` },
        marks: [
          Plot.ruleY([0], { stroke: muted, strokeDasharray: "2 3" }),
          Plot.line(data.pnl, { x: (d) => d[0], y: (d) => d[1], stroke: colour("--chart-3"), strokeWidth: 2 }),
          Plot.line(data.expected, { x: (d) => d[0], y: (d) => d[1], stroke: colour("--chart-1"), strokeWidth: 2 }),
        ],
      }),
    );
  };

  const bookSetsChart = (element, data) => {
    const muted = colour("--muted-foreground");
    const reach = Math.max(1, ...data.sets.map((s) => s.running.length));
    const stroke = (s) => (s.registered ? colour("--chart-1") : colour("--chart-2"));
    return Plot.plot(
      frame(element, {
        height: 170,
        x: { domain: [0, reach], label: "bets with a close", grid: false },
        y: { domain: [-data.axis, data.axis], label: null, tickFormat: (v) => `${v > 0 ? "+" : ""}${v}%` },
        marks: [
          Plot.ruleY([0], { stroke: muted, strokeDasharray: "2 3" }),
          ...data.sets.map((s) =>
            Plot.line(s.running, { x: (d) => d[0], y: (d) => d[1], stroke: stroke(s), strokeWidth: s.registered ? 2.5 : 1.5, strokeDasharray: s.registered ? null : "5 3", clip: true, title: () => s.name }),
          ),
        ],
      }),
    );
  };

  const drawers = { ev: evChart, cumulative: cumulativeChart, booksets: bookSetsChart };

  const draw = (root) => {
    for (const element of root.querySelectorAll(".chart[data-chart]")) {
      const data = JSON.parse(element.dataset.series);
      element.replaceChildren(drawers[element.dataset.chart](element, data));
    }
  };

  const drawAll = () => draw(document);
  document.addEventListener("DOMContentLoaded", drawAll);
  document.addEventListener("htmx:afterSettle", drawAll);
  let pending;
  window.addEventListener("resize", () => {
    clearTimeout(pending);
    pending = setTimeout(drawAll, 150);
  });
})();
