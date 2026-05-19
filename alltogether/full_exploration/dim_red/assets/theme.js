/* ============================================================
 * Theme bootstrapping — runs before Dash hydrates the DOM.
 * Sets [data-theme] on <html> from localStorage so we never
 * flash the wrong theme on reload.
 * ============================================================ */
(function () {
  try {
    var saved = localStorage.getItem('perch-viewer-theme');
    var theme = saved === 'dark' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', theme);
  } catch (e) {
    document.documentElement.setAttribute('data-theme', 'light');
  }
})();

/* Dash clientside_callbacks live here too. The theme toggle uses one
 * to update the <html> attribute AND localStorage in one round-trip. */
window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.theme = {
  apply: function (theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('perch-viewer-theme', theme); } catch (e) {}
    return theme; // echo back so Dash can mirror state if needed
  },
};
