"""
modules/athena/citation_graph_view.py

Draws a citation graph (backlog #60). Three outputs, none needing a library:

* **HTML** - one self-contained file: inline SVG, inline CSS, inline JS. No CDN,
  no network, so it opens offline from the exports folder and inside the web
  UI's frame. Arrows run from the citing paper to the cited one; node size is
  how often a paper is cited by your other documents; colour is its subject.
  Hover or click a paper to see what it cites and what cites it, click an arrow
  to see the reference entry that produced it, search by title or author, and
  switch between a force layout and a by-year timeline.
* **DOT** - Graphviz source, for anyone who wants to lay it out themselves.
* **JSON** - the graph itself (what ``GET /api/athena/citation-graph`` returns).

The layout is a pure function (``LAYOUT_JS``) kept apart from the DOM code so
it can be unit-tested under Node without a browser.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Layout - pure JS, no DOM. Fruchterman-Reingold with a fixed seed so the same
# graph always lands in the same place; optional by-year mode pins x to the year.
# ---------------------------------------------------------------------------

LAYOUT_JS = r"""
var CitationLayout = (function () {
  'use strict';
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // nodes: [{id, year?}]  links: [{source, target}] (ids)  ->  sets x, y on each node.
  function layout(nodes, links, o) {
    o = o || {};
    var W = o.width || 900, H = o.height || 600, mode = o.mode || 'force';
    var iters = o.iterations || 300, PAD = 60, rnd = mulberry32(o.seed || 7);
    var n = nodes.length;
    if (!n) return nodes;
    var idx = {};
    nodes.forEach(function (d, i) { idx[d.id] = i; });
    var P = nodes.map(function () {
      var a = rnd() * 6.2832, r = Math.sqrt(rnd()) * Math.min(W, H) * 0.35;
      return { x: W / 2 + Math.cos(a) * r, y: H / 2 + Math.sin(a) * r, dx: 0, dy: 0 };
    });
    var E = [];
    links.forEach(function (l) {
      var a = idx[l.source], b = idx[l.target];
      if (a !== undefined && b !== undefined && a !== b) E.push([a, b]);
    });
    var years = nodes.map(function (d) { return d.year; }).filter(function (y) { return typeof y === 'number'; });
    var byYear = mode === 'year' && years.length > 0;
    var y0 = years.length ? Math.min.apply(null, years) : 0;
    var y1 = years.length ? Math.max.apply(null, years) : 0;
    function tx(d) {
      if (typeof d.year !== 'number') return PAD * 0.7;                   // "year unknown" column, far left
      if (y1 === y0) return W / 2;
      return PAD * 1.8 + (d.year - y0) / (y1 - y0) * (W - PAD * 2.8);
    }
    var k = Math.sqrt(W * H / n) * 0.55;
    for (var it = 0; it < iters; it++) {
      var temp = (W / 8) * (1 - it / iters) + 0.5, i, j;
      for (i = 0; i < n; i++) { P[i].dx = 0; P[i].dy = 0; }
      for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
          var ddx = P[i].x - P[j].x, ddy = P[i].y - P[j].y, d2 = ddx * ddx + ddy * ddy;
          if (d2 < 0.01) { ddx = rnd() - 0.5; ddy = rnd() - 0.5; d2 = ddx * ddx + ddy * ddy + 0.01; }
          var d = Math.sqrt(d2), f = k * k / d2;                           // repulsion: k^2 / d
          P[i].dx += ddx * f; P[i].dy += ddy * f; P[j].dx -= ddx * f; P[j].dy -= ddy * f;
        }
      }
      E.forEach(function (e) {
        var a = P[e[0]], b = P[e[1]], ex = a.x - b.x, ey = a.y - b.y;
        var d = Math.sqrt(ex * ex + ey * ey) + 0.01, f = d / k;            // attraction: d^2 / k, as a ratio
        a.dx -= ex * f; a.dy -= ey * f; b.dx += ex * f; b.dy += ey * f;
      });
      for (i = 0; i < n; i++) {
        P[i].dx += (W / 2 - P[i].x) * 0.02 * k; P[i].dy += (H / 2 - P[i].y) * 0.02 * k;   // gentle gravity
        if (byYear) P[i].dx = P[i].dx * 0.1 + (tx(nodes[i]) - P[i].x) * 0.8;
        var len = Math.sqrt(P[i].dx * P[i].dx + P[i].dy * P[i].dy) || 1, m = Math.min(len, temp);
        P[i].x += P[i].dx / len * m; P[i].y += P[i].dy / len * m;
        P[i].x = Math.max(PAD, Math.min(W - PAD, P[i].x));
        P[i].y = Math.max(PAD, Math.min(H - PAD, P[i].y));
      }
    }
    if (!byYear && n > 1) {                                                // stretch to fill the frame
      var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      P.forEach(function (p) { minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y); });
      var sx = (maxX - minX) > 1 ? (W - 2 * PAD) / (maxX - minX) : 1, sy = (maxY - minY) > 1 ? (H - 2 * PAD) / (maxY - minY) : 1;
      var s = Math.min(sx, sy, 1.6);
      var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
      P.forEach(function (p) { p.x = W / 2 + (p.x - cx) * s; p.y = H / 2 + (p.y - cy) * s; });
    }
    nodes.forEach(function (d, i) { d.x = P[i].x; d.y = P[i].y; });
    return nodes;
  }
  return { layout: layout, mulberry32: mulberry32 };
})();
"""

# ---------------------------------------------------------------------------
# Page behaviour. Everything user-supplied is put in with textContent, never innerHTML.
# ---------------------------------------------------------------------------

UI_JS = r"""
(function () {
  'use strict';
  var DATA = JSON.parse(document.getElementById('graph-data').textContent);
  var NS = 'http://www.w3.org/2000/svg';
  var PALETTE = ['#4e79a7', '#f28e2b', '#59a14f', '#b07aa1', '#e15759', '#76b7b2', '#edc948', '#9c755f'];
  var byId = {}, subjects = [], state = { mode: 'force', showIsolated: false, selected: null, query: '' };
  DATA.nodes.forEach(function (n) { byId[n.id] = n; if (subjects.indexOf(n.subject) < 0) subjects.push(n.subject); });
  var hasLinks = DATA.links.length > 0;
  state.showIsolated = !hasLinks;
  var yearsKnown = DATA.nodes.filter(function (n) { return typeof n.year === 'number'; }).length;

  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function sv(tag, attrs) {
    var e = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    return e;
  }
  function short(s, n) { s = String(s || ''); return s.length > n ? s.slice(0, n - 1) + '\u2026' : s; }
  function colour(n) { var i = subjects.indexOf(n.subject); return i >= 0 && i < PALETTE.length ? PALETTE[i] : '#8a8f98'; }
  function radius(n) { return 6 + 3.2 * Math.sqrt(n.cited_by || 0); }

  var svg = $('g'), stage = $('stage');
  var defs = sv('defs');
  [['arrow', 'var(--edge)'], ['arrow-hi', 'var(--accent)']].forEach(function (m) {
    var mk = sv('marker', { id: m[0], viewBox: '0 0 10 10', refX: '10', refY: '5', markerWidth: '7', markerHeight: '7', orient: 'auto' });
    mk.appendChild(sv('path', { d: 'M0,0 L10,5 L0,10 z', fill: m[1] }));
    defs.appendChild(mk);
  });
  svg.appendChild(defs);
  var viewport = sv('g', { id: 'viewport' }), edgeLayer = sv('g'), nodeLayer = sv('g');
  viewport.appendChild(edgeLayer); viewport.appendChild(nodeLayer); svg.appendChild(viewport);
  var view = { k: 1, x: 0, y: 0 };
  function applyView() { viewport.setAttribute('transform', 'translate(' + view.x + ',' + view.y + ') scale(' + view.k + ')'); }

  var vis = [], visLinks = [], nodeEls = {}, edgeEls = [], recip = {};

  function visibleNodes() {
    var linked = {};
    DATA.links.forEach(function (l) { linked[l.source] = true; linked[l.target] = true; });
    return DATA.nodes.filter(function (n) { return state.showIsolated || linked[n.id]; }).map(function (n) {
      return Object.assign({}, n, { r: radius(n) });
    });
  }

  function edgePath(l) {
    var a = byId2[l.source], b = byId2[l.target];
    var dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) || 1, ux = dx / d, uy = dy / d;
    var x1 = a.x + ux * a.r, y1 = a.y + uy * a.r, x2 = b.x - ux * (b.r + 2), y2 = b.y - uy * (b.r + 2);
    if (recip[l.target + '>' + l.source]) {                              // A cites B and B cites A: bend both
      var cx = (x1 + x2) / 2 - uy * 22, cy = (y1 + y2) / 2 + ux * 22;
      return 'M' + x1 + ',' + y1 + ' Q' + cx + ',' + cy + ' ' + x2 + ',' + y2;
    }
    return 'M' + x1 + ',' + y1 + ' L' + x2 + ',' + y2;
  }
  var byId2 = {};

  function render() {
    var W = stage.clientWidth || 900, H = stage.clientHeight || 600;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    vis = visibleNodes(); byId2 = {}; vis.forEach(function (n) { byId2[n.id] = n; });
    visLinks = DATA.links.filter(function (l) { return byId2[l.source] && byId2[l.target]; });
    recip = {}; visLinks.forEach(function (l) { recip[l.source + '>' + l.target] = true; });
    CitationLayout.layout(vis, visLinks, { width: W, height: H, mode: state.mode });
    while (edgeLayer.firstChild) edgeLayer.removeChild(edgeLayer.firstChild);
    while (nodeLayer.firstChild) nodeLayer.removeChild(nodeLayer.firstChild);
    edgeEls = []; nodeEls = {};
    visLinks.forEach(function (l) {
      var p = sv('path', { 'class': 'edge' + (l.confidence < 0.9 ? ' weak' : ''), 'marker-end': 'url(#arrow)', d: edgePath(l) });
      var t = sv('title'); t.textContent = byId[l.source].label + ' \u2192 ' + byId[l.target].label + '  (' + l.method + ', ' + Math.round(l.confidence * 100) + '%)';
      p.appendChild(t);
      p.addEventListener('click', function (ev) { ev.stopPropagation(); showEdge(l); });
      edgeLayer.appendChild(p); edgeEls.push({ l: l, p: p });
    });
    vis.forEach(function (n) {
      var g = sv('g', { 'class': 'node', tabindex: '0', transform: 'translate(' + n.x + ',' + n.y + ')', role: 'button', 'aria-label': n.label });
      g.appendChild(sv('circle', { r: n.r, fill: colour(n) }));
      var showLabel = vis.length <= 30 || n.cited_by > 0;
      var tx = sv('text', { x: n.r + 4, y: 4, 'class': 'lbl' + (showLabel ? '' : ' hidden') }); tx.textContent = short(n.label, 34);
      tx._defaultHidden = !showLabel;
      g.appendChild(tx);
      var tt = sv('title'); tt.textContent = n.label + (n.year ? ' (' + n.year + ')' : '') + '\ncites ' + n.cites + ' \u00b7 cited by ' + n.cited_by;
      g.appendChild(tt);
      g.addEventListener('mouseenter', function () { highlight(n.id); });
      g.addEventListener('mouseleave', function () { highlight(state.selected); });
      g.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); select(n.id); } });
      g.addEventListener('click', function (ev) { ev.stopPropagation(); });
      drag(g, n);
      nodeLayer.appendChild(g); nodeEls[n.id] = g;
    });
    $('count').textContent = vis.length + ' of ' + DATA.nodes.length + ' documents shown';
    applyQuery(); highlight(state.selected);
  }

  function drag(g, n) {
    var moved = false, sx = 0, sy = 0;
    g.addEventListener('pointerdown', function (ev) {
      ev.stopPropagation(); moved = false; sx = ev.clientX; sy = ev.clientY;
      var move = function (e) {
        var dx = (e.clientX - sx) / view.k, dy = (e.clientY - sy) / view.k;
        if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
        sx = e.clientX; sy = e.clientY; n.x += dx; n.y += dy;
        g.setAttribute('transform', 'translate(' + n.x + ',' + n.y + ')');
        edgeEls.forEach(function (o) { o.p.setAttribute('d', edgePath(o.l)); });
      };
      var up = function () { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); if (!moved) select(n.id); };
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
    });
  }

  function neighbours(id) {
    var out = {}, inn = {};
    visLinks.forEach(function (l) { if (l.source === id) out[l.target] = true; if (l.target === id) inn[l.source] = true; });
    return { cites: out, citedBy: inn };
  }

  function highlight(id) {
    var nb = id ? neighbours(id) : null;
    Object.keys(nodeEls).forEach(function (k) {
      var on = !nb || k === id || nb.cites[k] || nb.citedBy[k];
      nodeEls[k].classList.toggle('dim', !on);
      nodeEls[k].classList.toggle('sel', k === id);
    });
    edgeEls.forEach(function (o) {
      var on = nb && (o.l.source === id || o.l.target === id);
      o.p.classList.toggle('dim', !!nb && !on); o.p.classList.toggle('hi', !!on);
      o.p.setAttribute('marker-end', on ? 'url(#arrow-hi)' : 'url(#arrow)');
    });
  }

  function listOf(ids, container) {
    if (!ids.length) { container.appendChild(el('li', { 'class': 'none' }, 'none')); return; }
    ids.forEach(function (i) {
      var li = el('li'), a = el('button', { type: 'button', 'class': 'link' }, short(byId[i].label, 80));
      a.addEventListener('click', function () { if (!byId2[i]) { state.showIsolated = true; $('iso').checked = true; render(); } select(i); });
      li.appendChild(a); container.appendChild(li);
    });
  }

  var STATUS = { ok: '', no_reference_section: 'No reference list found, so this paper can be cited but its own citations are unknown.',
    empty_reference_section: 'A reference list was found but could not be split into entries.',
    no_text: 'No text layer (scanned?), so its reference list could not be read.',
    unreadable: 'The file could not be opened.', missing: 'The file is no longer at its indexed path.' };

  function select(id) {
    state.selected = id; highlight(id);
    var box = $('detail'); box.textContent = '';
    if (!id) { box.appendChild(el('p', { 'class': 'hint' }, 'Click a paper to see what it cites and what cites it.')); return; }
    var n = byId[id];
    box.appendChild(el('h2', {}, n.label));
    var meta = [n.year, (n.authors || []).slice(0, 3).join(', '), n.arxiv_id ? 'arXiv:' + n.arxiv_id : '', n.doi ? 'doi:' + n.doi : ''].filter(Boolean).join(' \u00b7 ');
    if (meta) box.appendChild(el('p', { 'class': 'meta' }, meta));
    box.appendChild(el('p', { 'class': 'meta' }, n.file_name + ' \u00b7 ' + n.subject));
    if (STATUS[n.status]) box.appendChild(el('p', { 'class': 'warn' }, STATUS[n.status]));
    var cites = DATA.links.filter(function (l) { return l.source === id; }).map(function (l) { return l.target; });
    var by = DATA.links.filter(function (l) { return l.target === id; }).map(function (l) { return l.source; });
    box.appendChild(el('h3', {}, 'Cites (' + cites.length + (n.references ? ' of ' + n.references + ' listed references are in your library' : '') + ')'));
    var u1 = el('ul'); listOf(cites, u1); box.appendChild(u1);
    box.appendChild(el('h3', {}, 'Cited by (' + by.length + ')'));
    var u2 = el('ul'); listOf(by, u2); box.appendChild(u2);
  }

  function showEdge(l) {
    var box = $('detail'); box.textContent = '';
    box.appendChild(el('h2', {}, 'Why this arrow'));
    box.appendChild(el('p', {}, byId[l.source].label + ' cites ' + byId[l.target].label + '.'));
    box.appendChild(el('p', { 'class': 'meta' }, 'Matched by ' + l.method + ' \u00b7 ' + Math.round(l.confidence * 100) + '% confidence'));
    if (l.year_mismatch) box.appendChild(el('p', { 'class': 'warn' }, 'The years differ by more than two, so treat this as less certain.'));
    if (l.suspicious) box.appendChild(el('p', { 'class': 'warn' }, l.suspicious + '.'));
    box.appendChild(el('h3', {}, 'The reference entry'));
    box.appendChild(el('blockquote', {}, l.evidence));
  }

  function applyQuery() {
    var q = state.query.trim().toLowerCase();
    Object.keys(nodeEls).forEach(function (k) {
      var n = byId[k], hay = (n.label + ' ' + n.file_name + ' ' + (n.authors || []).join(' ') + ' ' + n.arxiv_id).toLowerCase();
      var hit = !q || hay.indexOf(q) >= 0;
      nodeEls[k].classList.toggle('nomatch', !hit);
      var t = nodeEls[k].querySelector('text'); if (t) t.classList.toggle('hidden', q ? !hit : !!t._defaultHidden);
    });
  }

  function fit() { view = { k: 1, x: 0, y: 0 }; applyView(); }

  // ---- static panels ----
  var s = DATA.stats;
  $('headline').textContent = DATA.nodes.length + ' documents \u00b7 ' + s.links + ' citations \u00b7 ' + s.connected_documents + ' linked' +
    (DATA.subject ? ' \u00b7 subject: ' + DATA.subject : '');
  var legend = $('legend');
  subjects.slice(0, PALETTE.length).forEach(function (sub, i) {
    var item = el('span', { 'class': 'key' }); var dot = el('i'); dot.style.background = PALETTE[i]; item.appendChild(dot); item.appendChild(document.createTextNode(sub)); legend.appendChild(item);
  });
  if (subjects.length > PALETTE.length) { var other = el('span', { 'class': 'key' }); var d2 = el('i'); d2.style.background = '#8a8f98'; other.appendChild(d2); other.appendChild(document.createTextNode('other')); legend.appendChild(other); }
  var sizeKey = el('span', { 'class': 'key plain' }, 'Bigger = cited more often \u00b7 dashed arrow = less certain match');
  legend.appendChild(sizeKey);

  var info = $('info');
  if (s.most_cited && s.most_cited.length) {
    info.appendChild(el('h3', {}, 'Most cited'));
    var ul = el('ul'); s.most_cited.forEach(function (m) { var li = el('li'), b = el('button', { type: 'button', 'class': 'link' }, short(m.label, 70) + ' (' + m.cited_by + ')'); b.addEventListener('click', function () { select(m.id); }); li.appendChild(b); ul.appendChild(li); }); info.appendChild(ul);
  }
  if (DATA.missing_but_cited && DATA.missing_but_cited.length) {
    info.appendChild(el('h3', {}, 'Cited by several of your papers, but not in your library'));
    var ul2 = el('ul'); DATA.missing_but_cited.slice(0, 8).forEach(function (m) { ul2.appendChild(el('li', {}, short(m.title || m.reference, 90) + (m.year ? ' (' + m.year + ')' : '') + ' \u2014 ' + m.cited_by_count + ' papers')); }); info.appendChild(ul2);
  }
  if (DATA.notes && DATA.notes.length) {
    info.appendChild(el('h3', {}, 'Notes'));
    var ul3 = el('ul'); DATA.notes.forEach(function (t) { ul3.appendChild(el('li', {}, t)); }); info.appendChild(ul3);
  }

  // ---- controls ----
  $('iso').checked = state.showIsolated;
  $('iso').addEventListener('change', function () { state.showIsolated = this.checked; render(); });
  $('mode-force').addEventListener('change', function () { state.mode = 'force'; render(); fit(); });
  $('mode-year').addEventListener('change', function () { state.mode = 'year'; render(); fit(); });
  if (yearsKnown < Math.max(2, DATA.nodes.length * 0.5)) { $('mode-year').disabled = true; $('mode-year-label').title = 'Needs publication years for at least half of the documents'; }
  $('q').addEventListener('input', function () { state.query = this.value; applyQuery(); });
  $('q').addEventListener('keydown', function (ev) {
    if (ev.key !== 'Enter') return;
    var q = state.query.trim().toLowerCase(); if (!q) return;
    var hit = DATA.nodes.filter(function (n) { return (n.label + ' ' + n.file_name + ' ' + (n.authors || []).join(' ')).toLowerCase().indexOf(q) >= 0; })[0];
    if (hit) { if (!byId2[hit.id]) { state.showIsolated = true; $('iso').checked = true; render(); } select(hit.id); }
  });
  $('fit').addEventListener('click', fit);
  var panned = false;
  svg.addEventListener('click', function () { if (panned) { panned = false; return; } select(null); });
  svg.addEventListener('wheel', function (ev) {
    ev.preventDefault();
    var r = svg.getBoundingClientRect(), W = svg.viewBox.baseVal.width || r.width, f = W / (r.width || W);
    var px = (ev.clientX - r.left) * f, py = (ev.clientY - r.top) * f, k2 = Math.max(0.2, Math.min(5, view.k * (ev.deltaY < 0 ? 1.15 : 1 / 1.15)));
    view.x = px - (px - view.x) * (k2 / view.k); view.y = py - (py - view.y) * (k2 / view.k); view.k = k2; applyView();
  }, { passive: false });
  svg.addEventListener('pointerdown', function (ev) {
    if (ev.target !== svg) return;
    var sx = ev.clientX, sy = ev.clientY;
    var move = function (e) { if (Math.abs(e.clientX - sx) + Math.abs(e.clientY - sy) > 2) panned = true; view.x += e.clientX - sx; view.y += e.clientY - sy; sx = e.clientX; sy = e.clientY; applyView(); };
    var up = function () { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  });

  if (!DATA.nodes.length) { $('empty').hidden = false; }
  render();
  if (DATA.focus && byId[DATA.focus]) select(DATA.focus); else select(null);
  window.__citationGraph = { state: state, select: select, render: render, data: DATA };   // for tests
})();
"""

CSS = r"""
:root { --bg:#fbfaf7; --panel:#ffffff; --text:#23262b; --muted:#6b7280; --line:#e3e0d8; --edge:#9aa1ab; --accent:#c1502e; }
@media (prefers-color-scheme: dark) { :root { --bg:#16191d; --panel:#1d2126; --text:#e6e8eb; --muted:#9aa3ad; --line:#2d3339; --edge:#6b7480; --accent:#ff8a5c; } }
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); display: flex; flex-direction: column; }
header { padding: 12px 16px; border-bottom: 1px solid var(--line); display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; }
header h1 { font-size: 16px; margin: 0; }
#headline, #count { color: var(--muted); font-size: 13px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin-left: auto; }
.controls input[type=search] { padding: 5px 9px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--text); min-width: 180px; }
.controls label { display: inline-flex; gap: 4px; align-items: center; cursor: pointer; }
.controls button, .link { font: inherit; }
.controls button { padding: 4px 10px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--text); cursor: pointer; }
main { flex: 1; min-height: 0; display: flex; }
#stage { flex: 1; min-width: 0; position: relative; }
#g { width: 100%; height: 100%; display: block; touch-action: none; }
aside { width: 340px; max-width: 42%; border-left: 1px solid var(--line); background: var(--panel); overflow: auto; padding: 12px 16px; }
aside h2 { font-size: 15px; margin: 0 0 4px; }
aside h3 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 14px 0 4px; }
aside ul { margin: 0; padding-left: 18px; } aside li { margin: 2px 0; }
.meta, .hint { color: var(--muted); margin: 2px 0; font-size: 13px; }
.warn { color: var(--accent); font-size: 13px; }
blockquote { margin: 4px 0; padding: 6px 10px; border-left: 3px solid var(--line); color: var(--muted); font-size: 13px; word-break: break-word; }
.link { background: none; border: 0; padding: 0; color: var(--accent); cursor: pointer; text-align: left; text-decoration: underline; }
.none { color: var(--muted); list-style: none; margin-left: -18px; }
#legend { padding: 6px 16px; display: flex; flex-wrap: wrap; gap: 4px 14px; border-bottom: 1px solid var(--line); font-size: 12px; color: var(--muted); }
.key { display: inline-flex; align-items: center; gap: 5px; } .key i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.edge { fill: none; stroke: var(--edge); stroke-width: 1.3; stroke-opacity: .75; cursor: pointer; }
.edge.weak { stroke-dasharray: 5 3; } .edge.hi { stroke: var(--accent); stroke-width: 2; stroke-opacity: 1; } .edge.dim { stroke-opacity: .08; }
.node { cursor: pointer; outline: none; } .node circle { stroke: var(--bg); stroke-width: 1.5; }
.node.sel circle, .node:focus circle { stroke: var(--accent); stroke-width: 3; }
.node.dim { opacity: .15; } .node.nomatch { opacity: .12; }
.lbl { font-size: 11px; fill: var(--text); pointer-events: none; paint-order: stroke; stroke: var(--bg); stroke-width: 3px; } .lbl.hidden { display: none; }
#empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--muted); text-align: center; padding: 24px; }
#empty[hidden] { display: none; }
@media (max-width: 760px) { main { flex-direction: column; } aside { width: auto; max-width: none; max-height: 40%; border-left: 0; border-top: 1px solid var(--line); } }
"""


def _json_for_script(obj: Any) -> str:
    """JSON that is safe inside <script>: no raw '<' (so no '</script>' or '<!--') and no U+2028/9."""
    text = json.dumps(obj, ensure_ascii=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def render_html(graph: dict, focus_id: Optional[str] = None) -> str:
    """The interactive page. *graph* is ``CitationGraph.to_dict()``; *focus_id* a node id to open on."""
    payload = dict(graph)
    payload["focus"] = focus_id
    title = "Citation graph" + (f" \u2014 {graph['subject']}" if graph.get("subject") else "")
    empty = ("No documents to draw yet. Index some papers and ask again."
             if not graph.get("nodes") else "")
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{_esc(title)}</title><style>{CSS}</style></head><body>\n"
        f"<header><h1>{_esc(title)}</h1><span id=\"headline\"></span><span id=\"count\"></span>"
        "<div class=\"controls\"><input id=\"q\" type=\"search\" placeholder=\"Find a paper\u2026\" aria-label=\"Find a paper\">"
        "<label id=\"mode-force-label\"><input type=\"radio\" name=\"mode\" id=\"mode-force\" checked> Force</label>"
        "<label id=\"mode-year-label\"><input type=\"radio\" name=\"mode\" id=\"mode-year\"> By year</label>"
        "<label><input type=\"checkbox\" id=\"iso\"> Show unlinked</label>"
        "<button type=\"button\" id=\"fit\">Reset view</button></div></header>\n"
        "<div id=\"legend\"></div>\n"
        "<main><div id=\"stage\"><svg id=\"g\" role=\"img\" aria-label=\"Citation graph: arrows point from a paper to the paper it cites\"></svg>"
        f"<div id=\"empty\" hidden>{_esc(empty)}</div></div>"
        "<aside><div id=\"detail\"></div><div id=\"info\"></div></aside></main>\n"
        f"<script type=\"application/json\" id=\"graph-data\">{_json_for_script(payload)}</script>\n"
        f"<script>{LAYOUT_JS}</script>\n<script>{UI_JS}</script>\n</body></html>\n"
    )


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


# ---------------------------------------------------------------------------
# DOT
# ---------------------------------------------------------------------------

def _dot_label(text: str, width: int = 34) -> str:
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    safe = [ln.replace("\\", "\\\\").replace('"', '\\"') for ln in lines[:4]]
    return "\\n".join(safe) + ("\\n..." if len(lines) > 4 else "")


def to_dot(graph: dict, include_isolated: bool = False) -> str:
    """Graphviz source; an arrow runs from the citing paper to the cited one."""
    linked = {l["source"] for l in graph["links"]} | {l["target"] for l in graph["links"]}
    out = ["digraph citations {", "  rankdir=LR;",
           '  node [shape=box, style="rounded", fontsize=10, fontname="Helvetica"];',
           '  edge [color="#8a8f98", arrowsize=0.7];']
    for n in graph["nodes"]:
        if not include_isolated and n["id"] not in linked:
            continue
        year = f"\\n({n['year']})" if n.get("year") else ""
        out.append(f'  {n["id"]} [label="{_dot_label(n["label"])}{year}"];')
    for l in graph["links"]:
        style = ' [style=dashed]' if l["confidence"] < 0.9 else ""
        out.append(f'  {l["source"]} -> {l["target"]}{style};')
    out.append("}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

_FORMATS = ("html", "json", "dot")


def write_graph_files(graph: dict, out_dir: str, stem: str, date: str,
                      formats: tuple[str, ...] = ("html", "json"),
                      focus_id: Optional[str] = None) -> list[str]:
    """Write the requested renderings to *out_dir*; returns the paths. Raises OSError if it cannot write."""
    slug = re.sub(r"[^a-z0-9]+", "-", (stem or "").lower()).strip("-")[:40] or "citation-graph"
    base = Path(out_dir) / f"{slug}-{date}"
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt not in _FORMATS:
            continue
        path = f"{base}.{fmt}"
        if fmt == "html":
            body = render_html(graph, focus_id)
        elif fmt == "dot":
            body = to_dot(graph)
        else:
            body = json.dumps(graph, indent=2, ensure_ascii=False)
        Path(path).write_text(body, encoding="utf-8")
        written.append(path)
    return written
