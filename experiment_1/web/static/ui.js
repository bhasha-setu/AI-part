// Tiny DOM helpers.
export function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === false || v == null) continue;
    if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'class') el.className = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  const add = k => {
    if (Array.isArray(k)) k.forEach(add);
    else if (k != null && k !== false) el.append(k.nodeType ? k : document.createTextNode(String(k)));
  };
  kids.forEach(add);
  return el;
}

export const clear = el => { while (el.firstChild) el.removeChild(el.firstChild); };

export function randomHex(n) {
  const a = new Uint8Array(Math.ceil(n / 2));
  (crypto && crypto.getRandomValues ? crypto.getRandomValues(a) : a.forEach((_, i) => { a[i] = Math.random() * 256; }));
  return Array.from(a, b => b.toString(16).padStart(2, '0')).join('').slice(0, n);
}

export const humanize = s => String(s || '').replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
