/* Exact in-browser inference for the trained scikit-learn SVC (RBF kernel).
 *
 *   f(x) = Σ_i  α_i y_i · exp(−γ‖z − sv_i‖²) + b        z = (x − μ) / σ   (StandardScaler)
 *   P(heatwave | x) = 1 − 1 / (1 + exp(−A·f(x) + B))    Platt scaling (libsvm probA/probB)
 *
 * Feature construction mirrors src/features.py and src/inference.py line for line.
 */
(function (global) {
  const DAY = 86400000;

  function doy(d) {
    const start = Date.UTC(d.getUTCFullYear(), 0, 0);
    return Math.min(Math.floor((d - start) / DAY), 365);
  }
  function parseDate(s) { const [y, m, dd] = s.split("-").map(Number); return new Date(Date.UTC(y, m - 1, dd)); }
  function addDays(d, n) { return new Date(d.getTime() + n * DAY); }
  function iso(d) { return d.toISOString().slice(0, 10); }

  class HeatwaveSVM {
    constructor(m) {
      this.m = m;
      this.F = m.features;
      this.nSV = m.support_vectors.length;
      this.dim = this.F.length;
      this.SV = new Float64Array(this.nSV * this.dim);
      m.support_vectors.forEach((sv, i) => sv.forEach((v, j) => (this.SV[i * this.dim + j] = v)));
    }
    normal(city, d) { return this.m.normals[city][doy(d) - 1]; }
    threshold(city) { return this.m.cities[city].coastal ? 37 : 40; }
    isHeatwave(city, tmax, dep) {
      const hw = (tmax >= this.threshold(city) && dep >= 4.5) || tmax >= 45;
      return { hw: hw ? 1 : 0, severe: hw && (dep > 6.4 || tmax >= 47) ? 1 : 0 };
    }
    /** inp: {tmax, tmax_1, tmax_2, tmin, app_tmax, rain7, wind_max}; today: "YYYY-MM-DD" */
    features(city, today, inp) {
      const t = parseDate(today), thr = this.threshold(city);
      const dep = inp.tmax - this.normal(city, t);
      const dep1 = inp.tmax_1 - this.normal(city, addDays(t, -1));
      const dep2 = inp.tmax_2 - this.normal(city, addDays(t, -2));
      const dep_mean3 = (dep + dep1 + dep2) / 3;
      const normal_next = this.normal(city, addDays(t, 1));
      const margin = (a) => { const p = normal_next + a; return Math.max(Math.min(p - thr, a - 4.5), p - 45); };
      return {
        margin: margin(dep), margin3: margin(dep_mean3), dep, dep_mean3,
        tmax_trend3: inp.tmax - inp.tmax_2, app_tmax: inp.app_tmax, dtr: inp.tmax - inp.tmin,
        rain7_log: Math.log1p(Math.max(inp.rain7, 0)), wind_max: inp.wind_max, normal_next,
        hw_today: this.isHeatwave(city, inp.tmax, dep).hw, coastal: this.m.cities[city].coastal,
      };
    }
    scale(fobj) { return this.F.map((k, j) => (fobj[k] - this.m.scaler_mean[j]) / this.m.scaler_scale[j]); }
    /** returns decision value, probability and per-support-vector contributions */
    predictVector(x) {
      const z = this.scale(x), g = this.m.gamma, D = this.dim;
      const contrib = new Float64Array(this.nSV), sim = new Float64Array(this.nSV);
      let f = this.m.intercept;
      for (let i = 0; i < this.nSV; i++) {
        let d2 = 0;
        for (let j = 0; j < D; j++) { const e = z[j] - this.SV[i * D + j]; d2 += e * e; }
        const k = Math.exp(-g * d2);
        sim[i] = k; contrib[i] = this.m.dual_coef[i] * k; f += contrib[i];
      }
      // libsvm stores the decision with the opposite sign to sklearn's decision_function
      const p = 1 - 1 / (1 + Math.exp(-this.m.probA * f + this.m.probB));
      // positive-class push vs negative-class pull
      let push = 0, pull = 0;
      for (let i = 0; i < this.nSV; i++) contrib[i] > 0 ? (push += contrib[i]) : (pull += contrib[i]);
      const order = Array.from(contrib.keys()).sort((a, b) => Math.abs(contrib[b]) - Math.abs(contrib[a])).slice(0, 8);
      return { f, p, z, push, pull, bias: this.m.intercept,
               top: order.map((i) => ({ i, c: contrib[i], k: sim[i], cls: this.m.dual_coef[i] > 0 ? 1 : 0 })) };
    }
    predict(city, today, inp) {
      const x = this.features(city, today, inp);
      return { x, ...this.predictVector(x), normalNext: x.normal_next,
               forecastFor: iso(addDays(parseDate(today), 1)) };
    }
    alert(p) {
      const t = this.m.threshold;
      if (p >= Math.max(0.5, 3 * t)) return { level: "Red", action: "Take action — heatwave very likely tomorrow" };
      if (p >= t) return { level: "Orange", action: "Be prepared — heatwave expected tomorrow" };
      if (p >= t / 2) return { level: "Yellow", action: "Be updated — elevated heat, keep watching" };
      return { level: "Green", action: "No action needed — heatwave unlikely" };
    }
  }
  global.HeatwaveSVM = HeatwaveSVM;
  global.hwDate = { parseDate, addDays, iso, doy };
  if (typeof module !== "undefined") module.exports = { HeatwaveSVM };
})(typeof window !== "undefined" ? window : globalThis);
