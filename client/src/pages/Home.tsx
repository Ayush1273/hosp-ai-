import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence, useTransform, useMotionValue } from "framer-motion";
import Lenis from "lenis";
import ReactMarkdown from "react-markdown";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  BedDouble,
  BrainCircuit,
  Check,
  Fan,
  Gauge,
  HeartPulse,
  Info,
  Loader2,
  RefreshCw,
  Sparkles,
  Stethoscope,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

// ================================================================
// DATA TYPES
// ================================================================

export interface OperationalInputState {
  er_arrivals: number;
  ambulance_arrivals: number;
  general_admissions: number;
  icu_admissions: number;
  icu_discharges: number;
  icu_transfers_in: number;
  icu_transfers_out: number;
  icu_occupied_beds: number;
  icu_available_beds: number;
  hdu_occupied_beds: number;
  general_occupied_beds: number;
  private_occupied_beds: number;
  ventilators_in_use: number;
  ventilators_available: number;
  icu_staff_available: number;
  staff_shortage: "YES" | "NO";
  elective_surgeries: number;
}

export const defaultInputs: OperationalInputState = {
  er_arrivals: 42,
  ambulance_arrivals: 11,
  general_admissions: 30,
  icu_admissions: 8,
  icu_discharges: 5,
  icu_transfers_in: 5,
  icu_transfers_out: 2,
  icu_occupied_beds: 43,
  icu_available_beds: 5,
  hdu_occupied_beds: 4,
  general_occupied_beds: 3,
  private_occupied_beds: 6,
  ventilators_in_use: 2,
  ventilators_available: 1,
  icu_staff_available: 12,
  staff_shortage: "YES",
  elective_surgeries: 3,
};

export interface PolicyResult {
  document: string;
  category: string;
  content: string;
  similarity: number;
}

export interface HospitalState {
  icu_beds_available: number;
  ventilators_available: number;
  icu_staff_available: number;
  staff_shortage: string;
}

export interface AnalysisResponse {
  current_occupancy: number;
  predicted_occupancy_24h: number;
  delta: number;
  risk_score: number;
  risk_level: "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  risk_components: Record<string, number>;
  risk_factors: string[];
  positive_trends: string[];
  forecast_change: number;
  hospital_state: HospitalState;
  retrieved_policies: PolicyResult[];
  ai_response: string;
  timestamp: string;
}

// ================================================================
// AMBIENT FLUID KEXSIO CANVAS BACKGROUND
// ================================================================

function FluidCanvas({ scrollYProgress }: { scrollYProgress?: any }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let currentScroll = 0;
    let unsub: (() => void) | undefined;
    if (scrollYProgress && typeof scrollYProgress.on === "function") {
      unsub = scrollYProgress.on("change", (latest: number) => {
        currentScroll = latest;
      });
    }

    let raf = 0;
    let width = 0;
    let height = 0;
    const dpr = window.devicePixelRatio || 1;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const THREAD_COUNT = 34;
    const POINTS_PER_LINE = 44;
    const colorPalette = [
      { r: 5, g: 150, b: 105 },
      { r: 16, g: 185, b: 129 },
      { r: 13, g: 148, b: 136 },
      { r: 4, g: 120, b: 87 },
      { r: 52, g: 211, b: 153 },
    ];

    const threads: any[] = [];
    for (let i = 0; i < THREAD_COUNT; i++) {
      const p = i / (THREAD_COUNT - 1);
      const colorIdx = Math.floor(p * (colorPalette.length - 1));
      const nextIdx = Math.min(colorIdx + 1, colorPalette.length - 1);
      const blend = p * (colorPalette.length - 1) - colorIdx;
      const c1 = colorPalette[colorIdx];
      const c2 = colorPalette[nextIdx];
      const r = Math.round(c1.r + (c2.r - c1.r) * blend);
      const g = Math.round(c1.g + (c2.g - c1.g) * blend);
      const b = Math.round(c1.b + (c2.b - c1.b) * blend);
      const distFromCenter = Math.abs(p - 0.5) * 2;
      const baseAlpha = 0.30 * (1 - Math.pow(distFromCenter, 1.4)) + 0.05;
      const lineWidth = 0.85 + (1 - distFromCenter) * 0.7;

      threads.push({
        p,
        r,
        g,
        b,
        baseAlpha,
        lineWidth,
        phaseOffset: p * Math.PI * 1.85,
        freqMultiplier: 0.85 + p * 0.3,
        speedMultiplier: 0.9 + p * 0.2,
      });
    }

    const updateGradients = () => {
      for (let i = 0; i < threads.length; i++) {
        const th = threads[i];
        const grad = ctx.createLinearGradient(0, 0, width, 0);
        grad.addColorStop(0, `rgba(${th.r},${th.g},${th.b},0)`);
        grad.addColorStop(0.15, `rgba(${th.r},${th.g},${th.b},${th.baseAlpha * 0.6})`);
        grad.addColorStop(0.5, `rgba(${th.r},${th.g},${th.b},${th.baseAlpha})`);
        grad.addColorStop(0.85, `rgba(${th.r},${th.g},${th.b},${th.baseAlpha * 0.6})`);
        grad.addColorStop(1, `rgba(${th.r},${th.g},${th.b},0)`);
        th.grad = grad;
      }
    };

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      updateGradients();
    };
    resize();
    window.addEventListener("resize", resize);

    const mouse = { x: width * 0.5, y: height * 0.5, active: false };
    const handleMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      mouse.active = true;
    };
    const handleLeave = () => {
      mouse.active = false;
    };
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseleave", handleLeave);

    updateGradients();

    const startTime = performance.now();

    const animate = (now: number) => {
      raf = requestAnimationFrame(animate);
      const elapsed = reduced ? 1.5 : (now - startTime) * 0.001;
      ctx.clearRect(0, 0, width, height);

      const centerY = height * 0.48;
      const baseWaveSpeed = reduced ? 0 : 0.38;

      for (let i = 0; i < THREAD_COUNT; i++) {
        const th = threads[i];
        const p = th.p;
        const centeredP = (p - 0.5) * 2;

        // Wave lines: alternate left and right horizontal drift with scroll
        const threadDir = i % 2 === 0 ? 1 : -1;
        const scrollShift = threadDir * currentScroll * 180;

        ctx.beginPath();
        for (let s = 0; s <= POINTS_PER_LINE; s++) {
          const u = s / POINTS_PER_LINE;
          const x = u * (width + 160) - 80 + scrollShift;
          const w1 =
            Math.sin(u * Math.PI * 2.2 + elapsed * baseWaveSpeed * th.speedMultiplier + th.phaseOffset) * 55;
          const w2 = Math.cos(u * Math.PI * 3.4 - elapsed * 0.26 + th.phaseOffset * 0.6) * 30;
          const spreadEnvelope = 35 + (Math.sin(Math.abs(u - 0.3) * Math.PI * 1.4) + 1.0) * 75;
          const threadSpreadY = centeredP * spreadEnvelope;

          let mouseDisplacement = 0;
          if (mouse.active) {
            const dx = x - mouse.x;
            const dy = centerY - mouse.y;
            const distSq = dx * dx + dy * dy;
            if (distSq < 280 * 280) {
              const dist = Math.sqrt(distSq);
              mouseDisplacement = (mouse.y - centerY) * (1 - dist / 280) * 0.28;
            }
          }

          const y = centerY + w1 + w2 + threadSpreadY + mouseDisplacement;
          if (s === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }

        ctx.strokeStyle = th.grad || "#10b981";
        ctx.lineWidth = th.lineWidth;
        ctx.stroke();
      }
    };

    raf = requestAnimationFrame(animate);
    return () => {
      cancelAnimationFrame(raf);
      unsub?.();
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseleave", handleLeave);
    };
  }, [scrollYProgress]);

  return <canvas ref={ref} className="fluid-canvas" aria-hidden="true" />;
}



// ================================================================
// STAT CARD COMPONENT
// ================================================================

function StatCard({
  label,
  value,
  suffix,
  hint,
  tone = "neutral",
  icon: Icon,
}: {
  label: string;
  value: string;
  suffix?: string;
  hint?: React.ReactNode;
  tone?: string;
  icon: any;
}) {
  return (
    <div className={`glass-card stat-card ${tone}`}>
      <div className="stat-top">
        <span className="eyebrow">{label}</span>
        <Icon size={16} />
      </div>
      <div className="stat-value">
        {value}
        {suffix && <small>{suffix}</small>}
      </div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

// ================================================================
// CLEAN FORMATTED AI BRIEFING RENDERER
// ================================================================

function FormattedAIResponse({ content }: { content: string }) {
  const sanitizedContent = useMemo(() => {
    if (!content) return "";
    // Clean up rogue orphan "#" lines (e.g. lines with just '#' or '# ' with nothing else)
    return content
      .replace(/^#\s*$/gm, "")
      .trim();
  }, [content]);

  if (!sanitizedContent) return null;

  return (
    <div className="ai-markdown-prose text-sm sm:text-[15px] leading-relaxed text-emerald-100/90 space-y-4">
      <ReactMarkdown
        components={{
          h1: ({ children }) => (
            <h3 className="text-sm sm:text-base font-bold uppercase tracking-wider text-emerald-400 mt-6 first:mt-0 mb-2.5 pb-2 border-b border-emerald-500/15">
              {children}
            </h3>
          ),
          h2: ({ children }) => (
            <h3 className="text-sm sm:text-base font-bold uppercase tracking-wider text-emerald-400 mt-6 first:mt-0 mb-2.5 pb-2 border-b border-emerald-500/15">
              {children}
            </h3>
          ),
          h3: ({ children }) => (
            <h4 className="text-sm sm:text-[15px] font-bold uppercase tracking-wider text-emerald-300 mt-5 first:mt-0 mb-2 pb-1.5 border-b border-emerald-500/15">
              {children}
            </h4>
          ),
          h4: ({ children }) => (
            <h5 className="text-xs sm:text-sm font-bold uppercase tracking-wider text-emerald-300 mt-4 mb-2">
              {children}
            </h5>
          ),
          p: ({ children }) => (
            <p className="text-sm sm:text-[15px] leading-relaxed text-emerald-100/90 mb-3 last:mb-0">
              {children}
            </p>
          ),
          ul: ({ children }) => (
            <ul className="space-y-2 mb-3.5 pl-5 list-disc marker:text-emerald-400 text-sm sm:text-[15px] text-emerald-100/90">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="space-y-2 mb-3.5 pl-5 list-decimal marker:text-emerald-400 text-sm sm:text-[15px] text-emerald-100/90">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed pl-1">
              {children}
            </li>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-emerald-200">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="italic text-emerald-200/90">
              {children}
            </em>
          ),
          hr: () => (
            <hr className="my-5 border-emerald-500/15" />
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-emerald-400/40 pl-3.5 py-2 my-3 bg-emerald-950/30 rounded-r text-sm italic text-emerald-200/90">
              {children}
            </blockquote>
          ),
          code: ({ children }) => (
            <code className="px-1.5 py-0.5 rounded bg-emerald-950/60 border border-emerald-500/20 text-emerald-300 font-mono text-xs sm:text-[13px]">
              {children}
            </code>
          ),
        }}
      >
        {sanitizedContent}
      </ReactMarkdown>
    </div>
  );
}

// ================================================================
// MAIN HOSP-AI COMMAND APPLICATION
// ================================================================

export default function Home() {
  const [inputs, setInputs] = useState<OperationalInputState>(defaultInputs);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  const lenisRef = useRef<Lenis | null>(null);

  const smoothScrollY = useMotionValue(0);

  // Initialize Lenis luxury smooth scroll and synchronously drive Framer Motion
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.1,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      orientation: "vertical",
      gestureOrientation: "vertical",
      smoothWheel: true,
      syncTouch: false,
    });
    lenisRef.current = lenis;

    // Direct synchronous tick in Lenis animation frame:
    // Eliminates browser scroll-event lag, layout thrashing, and micro-stutter
    const handleScroll = (e: any) => {
      smoothScrollY.set(e.scroll);
    };
    lenis.on("scroll", handleScroll);

    function raf(time: number) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }
    const reqId = requestAnimationFrame(raf);

    smoothScrollY.set(window.scrollY || 0);

    const onNativeScroll = () => {
      if (!lenisRef.current) {
        smoothScrollY.set(window.scrollY);
      }
    };
    window.addEventListener("scroll", onNativeScroll, { passive: true });

    return () => {
      cancelAnimationFrame(reqId);
      window.removeEventListener("scroll", onNativeScroll);
      lenis.off("scroll", handleScroll);
      lenis.destroy();
      lenisRef.current = null;
    };
  }, [smoothScrollY]);

  // Multi-line alternating horizontal motion on scroll:
  // Directly driven by Lenis RAF for butter-smooth 60/120fps motion without jank
  // Line 1 (Heading 1: "Predict ICU capacity pressure."): slides completely LEFT
  const line1X = useTransform(smoothScrollY, [0, 480], ["0vw", "-120vw"]);
  const line1Opacity = useTransform(smoothScrollY, [0, 380], [1, 0]);

  // Line 2 (Heading 2: "Before critical thresholds are breached."): slides completely RIGHT
  const line2X = useTransform(smoothScrollY, [0, 480], ["0vw", "120vw"]);
  const line2Opacity = useTransform(smoothScrollY, [0, 380], [1, 0]);

  // Line 3 (Subtitle description): slides completely LEFT
  const line3X = useTransform(smoothScrollY, [0, 480], ["0vw", "-120vw"]);
  const line3Opacity = useTransform(smoothScrollY, [0, 380], [1, 0]);

  // Line 4 (CTA Button): slides completely RIGHT
  const line4X = useTransform(smoothScrollY, [0, 480], ["0vw", "120vw"]);
  const line4Opacity = useTransform(smoothScrollY, [0, 380], [1, 0]);

  // Line 5 (Telemetry Specs Strip): slides completely LEFT
  const line5X = useTransform(smoothScrollY, [0, 480], ["0vw", "-120vw"]);
  const line5Opacity = useTransform(smoothScrollY, [0, 380], [1, 0]);

  const canvasScrollProgress = useTransform(smoothScrollY, [0, 600], [0, 1]);

  const now = useMemo(() => new Date(), []);

  const updateNumber = (key: keyof OperationalInputState, val: string) => {
    const num = Math.max(0, parseInt(val, 10) || 0);
    setInputs((prev) => ({ ...prev, [key]: num }));
  };

  const handleReset = () => {
    setInputs(defaultInputs);
    toast.info("Operational inputs reset to baseline defaults.");
  };

  const scrollToSnapshot = () => {
    if (lenisRef.current) {
      lenisRef.current.scrollTo("#snapshot-section", { offset: -30 });
    } else {
      document.getElementById("snapshot-section")?.scrollIntoView({ behavior: "smooth" });
    }
  };

  const scrollToTop = () => {
    if (lenisRef.current) {
      lenisRef.current.scrollTo(0, { duration: 1.25 });
    } else {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const scrollToResults = () => {
    const tryScroll = (attempts = 0) => {
      const elem = document.getElementById("results-section");
      if (elem) {
        if (lenisRef.current) {
          lenisRef.current.scrollTo(elem, { offset: -90, duration: 1.25 });
        } else {
          elem.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      } else if (attempts < 8) {
        setTimeout(() => tryScroll(attempts + 1), 50);
      }
    };
    setTimeout(() => tryScroll(), 60);
  };

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || `Analysis request failed with status ${response.status}`);
      }

      const data: AnalysisResponse = await response.json();
      setResult(data);
      toast.success("Condition analysis & risk evaluation completed.");

      // Smooth scroll to the top of the results section
      scrollToResults();
    } catch (err: any) {
      console.error("[Analysis Error]", err);
      toast.error(err.message || "Unable to complete analysis. Check server connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <FluidCanvas scrollYProgress={canvasScrollProgress} />
      <div className="ambient-grid" />
      <div className="app-shell relative z-10">
        {/* FLOATING CLINICAL TOPBAR */}
        <header className="topbar">
          <div
            className="brand cursor-pointer select-none"
            onClick={scrollToTop}
            title="Return to Top"
          >
            <div className="brand-mark">
              <img
                src="/assets/logo.png"
                alt="HOSP-AI Logo"
                className="w-4 h-4 object-contain"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = "none";
                }}
              />
            </div>
            <div>
              <strong>
                HOSP<span className="text-emerald-400">-AI</span> COMMAND
              </strong>
              <small>Hospital Operational Decision Support</small>
            </div>
          </div>

          <div className="flex-1" />

          <div className="top-actions flex items-center gap-3">
            <span className="text-xs text-emerald-300/80 hidden md:inline font-medium">
              {now.toLocaleDateString(undefined, {
                month: "short",
                day: "numeric",
                year: "numeric",
              })}
            </span>
            <button
              onClick={scrollToSnapshot}
              className="text-xs font-semibold text-emerald-400 hover:text-emerald-300 px-3.5 py-1.5 rounded-full border border-emerald-500/25 bg-emerald-950/40 transition-all hover:bg-emerald-900/50"
            >
              Snapshot ↓
            </button>
          </div>
        </header>

        <main className="relative z-10">
          {/* ============================================================== */}
          {/* MODERN MINIMAL HERO SECTION (LINES PART LEFT & RIGHT ON SCROLL) */}
          {/* ============================================================== */}
          <section className="hero-minimal">
            {/* HEADING (LINE 1 SLIDES LEFT, LINE 2 SLIDES RIGHT) */}
            <div className="hero-h1 max-w-4xl mx-auto flex flex-col items-center">
              {/* LINE 1: SLIDES LEFT ON SCROLL */}
              <motion.div
                style={{ x: line1X, opacity: line1Opacity }}
                className="hero-motion-line text-center"
              >
                Predict ICU capacity pressure.
              </motion.div>

              {/* LINE 2: SLIDES RIGHT ON SCROLL */}
              <motion.div
                style={{ x: line2X, opacity: line2Opacity }}
                className="hero-motion-line text-center mint-line mt-1"
              >
                Before critical thresholds are breached.
              </motion.div>
            </div>

            {/* LINE 3 (SUBTITLE): SLIDES LEFT ON SCROLL */}
            <motion.p
              style={{ x: line3X, opacity: line3Opacity }}
              className="hero-subtitle hero-motion-line mt-4"
            >
              Deterministic 24-hour predictive census velocity, arithmetic risk scoring, and policy-grounded
              guidance engineered for hospital leadership and bed management huddles.
            </motion.p>

            {/* INTERACTIVE CONTROLS */}
            <div className="flex flex-col items-center gap-5 mt-2">
              {/* LINE 4 (CTA BUTTON): SLIDES RIGHT ON SCROLL */}
              <motion.div
                style={{ x: line4X, opacity: line4Opacity }}
                className="hero-motion-line"
              >
                <button onClick={scrollToSnapshot} className="cta-island group">
                  <span>Configure Operational Conditions</span>
                  <span className="cta-icon-nest">
                    <ArrowDown size={14} />
                  </span>
                </button>
              </motion.div>

              {/* LINE 5 (TELEMETRY SPECS STRIP): SLIDES LEFT ON SCROLL */}
              <motion.div
                style={{ x: line5X, opacity: line5Opacity }}
                className="hero-motion-line flex flex-wrap justify-center items-center gap-2.5 sm:gap-3.5 text-xs text-emerald-400/80 uppercase tracking-wider font-semibold pt-2"
              >
                <span className="px-3 py-1.5 rounded-md bg-emerald-950/50 border border-emerald-500/20">
                  39-Feature XGBoost ML
                </span>
                <span className="text-emerald-500/40">·</span>
                <span className="px-3 py-1.5 rounded-md bg-emerald-950/50 border border-emerald-500/20">
                  Deterministic 0–10 Risk Engine
                </span>
                <span className="text-emerald-500/40">·</span>
                <span className="px-3 py-1.5 rounded-md bg-emerald-950/50 border border-emerald-500/20">
                  RAG Policy Retrieval
                </span>
                <span className="text-emerald-500/40">·</span>
                <span className="px-3 py-1.5 rounded-md bg-emerald-950/50 border border-emerald-500/20">
                  Gemini 3.6 Flash Grounding
                </span>
              </motion.div>
            </div>
          </section>

          {/* ============================================================== */}
          {/* OPERATIONAL CONDITIONS SNAPSHOT (SMOOTH SCROLL REVEAL) */}
          {/* ============================================================== */}
          <section
            id="snapshot-section"
            className="relative z-20 max-w-5xl mx-auto mt-6 mb-16 scroll-mt-24"
          >
            <div className="bezel-outer">
              <div className="bezel-inner p-6 sm:p-8">
              {/* ENCLOSURE HEADER */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 mb-6 border-b border-emerald-500/15 gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="eyebrow mint">Operational Conditions Snapshot</span>
                  </div>
                  <h2 className="text-2xl font-bold text-white mt-1">
                    Configure Current Census & Flow Telemetry
                  </h2>
                  <p className="text-sm text-emerald-200/80 mt-1">
                    18 operational indicators feeding the 39-feature machine learning model
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleReset}
                  className="self-start sm:self-auto text-xs sm:text-sm font-semibold text-emerald-400 hover:text-emerald-300 flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-950/50 border border-emerald-500/25 transition-all hover:bg-emerald-900/50"
                  title="Reset to baseline values"
                >
                  <RefreshCw size={14} /> Reset Baseline
                </button>
              </div>

              {/* 2-COLUMN, 2-ROW ENCLOSED CONTROL PANEL */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-5">
                {/* 1. EMERGENCY & INTAKE INFLOW */}
                <div className="form-group-box">
                  <div className="group-title-clean">
                    <Activity size={14} className="text-emerald-400" />
                    <span>Intake Inflow</span>
                  </div>
                  <div className="field-grid-clean">
                    <label>
                      <span>ER Arrivals</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.er_arrivals}
                        onChange={(e) => updateNumber("er_arrivals", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>Ambulance</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.ambulance_arrivals}
                        onChange={(e) => updateNumber("ambulance_arrivals", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>General In</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.general_admissions}
                        onChange={(e) => updateNumber("general_admissions", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>ICU Direct In</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.icu_admissions}
                        onChange={(e) => updateNumber("icu_admissions", e.target.value)}
                      />
                    </label>
                  </div>
                </div>

                {/* 2. ICU PATIENT FLOW */}
                <div className="form-group-box">
                  <div className="group-title-clean">
                    <HeartPulse size={14} className="text-emerald-400" />
                    <span>ICU Patient Flow</span>
                  </div>
                  <div className="field-grid-clean">
                    <label>
                      <span>Discharges</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.icu_discharges}
                        onChange={(e) => updateNumber("icu_discharges", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>Transfers In</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.icu_transfers_in}
                        onChange={(e) => updateNumber("icu_transfers_in", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>Transfers Out</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.icu_transfers_out}
                        onChange={(e) => updateNumber("icu_transfers_out", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>Occupied Beds</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.icu_occupied_beds}
                        onChange={(e) => updateNumber("icu_occupied_beds", e.target.value)}
                      />
                    </label>
                  </div>
                </div>

                {/* 3. BED INVENTORY */}
                <div className="form-group-box">
                  <div className="group-title-clean">
                    <BedDouble size={14} className="text-emerald-400" />
                    <span>Bed Inventory</span>
                  </div>
                  <div className="field-grid-clean">
                    <label>
                      <span>ICU Available</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.icu_available_beds}
                        onChange={(e) => updateNumber("icu_available_beds", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>HDU Occupied</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.hdu_occupied_beds}
                        onChange={(e) => updateNumber("hdu_occupied_beds", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>General Occ.</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.general_occupied_beds}
                        onChange={(e) => updateNumber("general_occupied_beds", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>Private Occ.</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.private_occupied_beds}
                        onChange={(e) => updateNumber("private_occupied_beds", e.target.value)}
                      />
                    </label>
                  </div>
                </div>

                {/* 4. CRITICAL RESOURCES & SHORTAGE */}
                <div className="form-group-box">
                  <div className="group-title-clean">
                    <Fan size={14} className="text-emerald-400" />
                    <span>Critical Resources</span>
                  </div>
                  <div className="field-grid-clean">
                    <label>
                      <span>Vents in Use</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.ventilators_in_use}
                        onChange={(e) => updateNumber("ventilators_in_use", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>Vents Reserve</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.ventilators_available}
                        onChange={(e) => updateNumber("ventilators_available", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>ICU Staff</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.icu_staff_available}
                        onChange={(e) => updateNumber("icu_staff_available", e.target.value)}
                      />
                    </label>
                    <label>
                      <span>Surgeries</span>
                      <Input
                        type="number"
                        min="0"
                        value={inputs.elective_surgeries}
                        onChange={(e) => updateNumber("elective_surgeries", e.target.value)}
                      />
                    </label>
                  </div>

                  {/* STAFF SHORTAGE TOGGLE */}
                  <div className="mt-3 pt-2.5 border-t border-emerald-500/15 flex items-center justify-between">
                    <span className="text-xs sm:text-sm font-semibold text-emerald-200">
                      Staff Shortage:
                    </span>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => setInputs((p) => ({ ...p, staff_shortage: "YES" }))}
                        className={`px-3 py-1 rounded text-xs font-bold transition-all ${
                          inputs.staff_shortage === "YES"
                            ? "bg-amber-500 text-slate-950 shadow-sm"
                            : "bg-emerald-950/50 text-emerald-400/70 border border-emerald-500/25"
                        }`}
                      >
                        YES
                      </button>
                      <button
                        type="button"
                        onClick={() => setInputs((p) => ({ ...p, staff_shortage: "NO" }))}
                        className={`px-3 py-1 rounded text-xs font-bold transition-all ${
                          inputs.staff_shortage === "NO"
                            ? "bg-emerald-500 text-slate-950 shadow-sm"
                            : "bg-emerald-950/50 text-emerald-400/70 border border-emerald-500/25"
                        }`}
                      >
                        NO
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* CENTERED ACTION BUTTON */}
              <div className="text-center pt-3">
                <Button
                  className={`analyze-btn group ${loading ? "is-loading" : ""}`}
                  onClick={handleAnalyze}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 size={16} className="animate-spin text-white shrink-0" />
                      <span>Running Predictive Pipeline…</span>
                    </>
                  ) : (
                    <>
                      <Sparkles size={16} className="text-emerald-200 shrink-0" />
                      <span>ANALYZE CONDITIONS</span>
                      <ArrowRight size={14} className="text-white/80 shrink-0 transition-transform group-hover:translate-x-1" />
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </section>

        {/* ============================================================== */}
        {/* RESULTS SECTION (ONLY APPEARS AFTER ANALYSIS) */}
        {/* ============================================================== */}
        <AnimatePresence>
          {result && (
            <motion.section
              id="results-section"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 30 }}
              transition={{ duration: 0.75, ease: [0.16, 1, 0.3, 1] }}
              className="max-w-5xl mx-auto space-y-8 scroll-mt-24 mb-16"
            >
              {/* CAPACITY OUTLOOK CARDS */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <span className="eyebrow mint">Capacity Outlook & Decision Signal</span>
                    <h3 className="text-2xl font-bold text-white mt-1">24-Hour Forecast & Risk Level</h3>
                  </div>
                  <span className={`status-badge ${result.risk_level.toLowerCase()}`}>
                    {result.risk_level} RISK
                  </span>
                </div>

                <div className="stats-grid">
                  <StatCard
                    label="Current ICU Occupancy"
                    value={`${result.current_occupancy.toFixed(1)}`}
                    suffix="%"
                    hint={
                      <>
                        <BedDouble size={14} /> Real-time census
                      </>
                    }
                    tone="mint"
                    icon={BedDouble}
                  />
                  <StatCard
                    label="24h Predicted Occupancy"
                    value={`${result.predicted_occupancy_24h.toFixed(1)}`}
                    suffix="%"
                    hint={
                      result.delta >= 0 ? (
                        <>
                          <TrendingUp size={14} className="text-amber-400" /> +{result.delta.toFixed(1)} pp shift
                        </>
                      ) : (
                        <>
                          <TrendingDown size={14} className="text-emerald-400" /> {result.delta.toFixed(1)} pp shift
                        </>
                      )
                    }
                    tone={result.delta > 0 ? "orange" : "neutral"}
                    icon={Activity}
                  />
                  <StatCard
                    label="Operational Risk Score"
                    value={result.risk_score.toFixed(1)}
                    suffix=" / 10"
                    hint="Deterministic arithmetic"
                    tone={result.risk_level.toLowerCase()}
                    icon={Gauge}
                  />
                  <StatCard
                    label="Risk Classification"
                    value={result.risk_level}
                    hint="Human review required"
                    tone={result.risk_level.toLowerCase()}
                    icon={AlertTriangle}
                  />
                </div>
              </div>

              {/* CURRENT HOSPITAL STATE */}
              <div className="glass-card p-6">
                <div className="flex items-center justify-between mb-4">
                  <span className="eyebrow">Hospital State Telemetry</span>
                  <span className="text-xs sm:text-sm text-emerald-400/80 font-medium">Critical Care Resources</span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="p-4 rounded-xl bg-[#0a1711] border border-emerald-500/25 text-center shadow-md">
                    <small className="text-xs text-emerald-400/80 uppercase tracking-wider font-semibold block mb-1">
                      ICU Available
                    </small>
                    <strong className="text-2xl sm:text-3xl font-bold text-emerald-200">
                      {result.hospital_state.icu_beds_available}
                    </strong>
                    <span className="text-xs text-emerald-500/80 block mt-1 font-medium">beds ready</span>
                  </div>
                  <div className="p-4 rounded-xl bg-[#0a1711] border border-emerald-500/25 text-center shadow-md">
                    <small className="text-xs text-emerald-400/80 uppercase tracking-wider font-semibold block mb-1">
                      Ventilators
                    </small>
                    <strong className="text-2xl sm:text-3xl font-bold text-emerald-200">
                      {result.hospital_state.ventilators_available}
                    </strong>
                    <span className="text-xs text-emerald-500/80 block mt-1 font-medium">in reserve</span>
                  </div>
                  <div className="p-4 rounded-xl bg-[#0a1711] border border-emerald-500/25 text-center shadow-md">
                    <small className="text-xs text-emerald-400/80 uppercase tracking-wider font-semibold block mb-1">
                      ICU Staff
                    </small>
                    <strong className="text-2xl sm:text-3xl font-bold text-emerald-200">
                      {result.hospital_state.icu_staff_available}
                    </strong>
                    <span className="text-xs text-emerald-500/80 block mt-1 font-medium">nurses on duty</span>
                  </div>
                  <div className="p-4 rounded-xl bg-[#0a1711] border border-emerald-500/25 text-center shadow-md">
                    <small className="text-xs text-emerald-400/80 uppercase tracking-wider font-semibold block mb-1">
                      Staff Shortage
                    </small>
                    <strong
                      className={`text-2xl sm:text-3xl font-bold ${
                        result.hospital_state.staff_shortage === "YES" ? "text-amber-400" : "text-emerald-300"
                      }`}
                    >
                      {result.hospital_state.staff_shortage}
                    </strong>
                    <span className="text-xs text-emerald-500/80 block mt-1 font-medium">escalation</span>
                  </div>
                </div>
              </div>

              {/* KEY RISK FACTORS & FORECAST TREND */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* BOTTLENECK ALERTS */}
                <div className="glass-card p-6">
                  <div className="flex items-center justify-between mb-3">
                    <span className="eyebrow">Triggered Risk Factors</span>
                    <span className="text-xs sm:text-sm text-emerald-400 font-semibold">
                      {result.risk_factors.length} identified
                    </span>
                  </div>
                  {result.risk_factors.length > 0 ? (
                    result.risk_factors.map((factor, i) => (
                      <div
                        key={factor}
                        className={`alert-chip ${i === 0 && result.risk_score >= 6 ? "critical" : ""}`}
                      >
                        <AlertTriangle size={17} />
                        <span>{factor}</span>
                      </div>
                    ))
                  ) : (
                    <div className="clear-chip">
                      <Check size={17} /> No major operational bottlenecks triggered.
                    </div>
                  )}
                </div>

                {/* 24-HOUR TRAJECTORY */}
                <div className="glass-card p-6">
                  <div className="flex items-center justify-between mb-3">
                    <span className="eyebrow">24-Hour Predictive Velocity</span>
                    <span className="text-xs sm:text-sm text-emerald-400 font-bold">
                      {result.forecast_change >= 0 ? `+${result.forecast_change} pp` : `${result.forecast_change} pp`}
                    </span>
                  </div>
                  {result.positive_trends.length > 0 ? (
                    result.positive_trends.map((trend, i) => (
                      <div key={i} className="clear-chip mb-2">
                        <Check size={16} /> {trend}
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-emerald-100/90 leading-relaxed mt-2">
                      {result.forecast_change > 0
                        ? "The 24-hour forecast indicates an increase in ICU occupancy pressure."
                        : result.forecast_change < 0
                        ? "The 24-hour forecast indicates a decrease in ICU occupancy pressure."
                        : "The 24-hour forecast indicates stable ICU census over the next day."}
                    </p>
                  )}
                </div>
              </div>

              {/* AI GROUNDED DECISION SUPPORT */}
              <div className="glass-card p-6 sm:p-8">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 mb-5 border-b border-emerald-500/15 gap-2">
                  <div>
                    <span className="eyebrow mint">AI Decision Support Briefing</span>
                    <h3 className="text-xl sm:text-2xl font-bold text-white mt-1">
                      Synthesized Guidance Grounded in Hospital SOPs
                    </h3>
                  </div>
                  <span className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-950/50 border border-emerald-500/25 text-xs font-semibold text-emerald-300">
                    <BrainCircuit size={14} /> Gemini 3.6 Flash
                  </span>
                </div>

                <FormattedAIResponse content={result.ai_response} />
              </div>
            </motion.section>
          )}
        </AnimatePresence>
      </main>

      {/* MINIMAL CLINICAL FOOTER */}
      <footer>
        <span>
          <Stethoscope size={13} /> Administrative resource-management only · Not clinical diagnostic advice
        </span>
        <span>
          <Info size={13} /> Human hospital administrator review required for every signal
        </span>
      </footer>
    </div>
  </>
);
}
