---
name: echarts-reporting
description: >-
  Add Apache ECharts option JSON for bar, line, pie, and combo charts when
  analytics answers benefit from visualization. Use after warehouse-backed numbers.
---

# ECharts reporting for analyst answers

## When the user asks to chart the previous answer

"Render this in a chart", "graph it", or "plot it" means a new view of figures already given, not a second copy of the write-up.

- Reply with a `##` title and exactly one `echarts-json` block.
- Use a bar chart for a ranking. Put the threshold in the title when they named one.
- Do not repeat the table, the names in prose, observations, or recommendations.

## When to chart

Add a chart when **all** of the following hold:

- The answer includes **structured numeric data** from the warehouse (not hypothetical).
- A visual pattern helps: comparison across categories, share of total, or trend over time.
- There are at least **two** data points (or one time series with multiple periods).

Skip charts for single scalars with no breakdown (unless the user explicitly asks for a chart).

## Chart type selection

| Pattern | Chart |
|---------|--------|
| Share of whole, ≤8 slices | `pie` |
| Rank or compare categories | `bar` |
| Time or ordered sequence | `line` |
| Volume + rate | `bar` + second axis or two small multiples |

Use **multiple** chart blocks when one view cannot show the story (e.g. trend + breakdown).

## Output format (after the narrative)

Place one or more fenced blocks **after** the prose. The UI parses `echarts-json` and passes the object to ECharts `setOption`.

### Single chart

```echarts-json
{
  "title": { "text": "Revenue by category (last 30 days)" },
  "tooltip": { "trigger": "axis" },
  "xAxis": { "type": "category", "data": ["A", "B", "C"] },
  "yAxis": { "type": "value" },
  "series": [{ "type": "bar", "data": [1200, 980, 640] }]
}
```

### Multiple charts

Use a JSON **array** of option objects in one block:

```echarts-json
[
  { "title": { "text": "Orders by month" }, "xAxis": { "type": "category", "data": ["Jan", "Feb"] }, "yAxis": { "type": "value" }, "series": [{ "type": "line", "data": [100, 120] }] },
  { "title": { "text": "Payment mix" }, "series": [{ "type": "pie", "radius": "60%", "data": [{ "name": "Card", "value": 70 }, { "name": "PayPal", "value": 30 }] }] }
]
```

## Rules

- Labels and titles must match the narrative time range and units.
- Do not embed chart JSON inside the main paragraph; keep narrative human-readable.
- Do not mention ECharts or "chart block" in the narrative — just present the analysis; the block is for the renderer.
