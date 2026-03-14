import { useState, useRef, useEffect, useCallback } from "react";
import { jsPDF } from "jspdf";

// ─── CONFIG ──────────────────────────────────────────────────────────────────
// Set VITE_GEMINI_API_KEY in your .env file — never hardcode secrets
const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_API_KEY;
const GEMINI_MODEL = "gemini-2.5-flash";
const GEMINI_URL = GEMINI_API_KEY
  ? `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`
  : null;

const PHASES = ["Introduction", "Technical", "Problem Solving", "Managerial", "Closing"];
const QUESTIONS_PER_PHASE = 1; // 5 total to reduce API usage
const MIN_QUESTIONS_FOR_SCORE = 4;
const SESSION_STORAGE_KEY = "interviewsim_session_v1";
const THEME_STORAGE_KEY = "interviewsim_theme_v1";
const USE_AI_EVALUATION = true;
const USE_AI_FINAL_SCORECARD = true;

const DEFAULT_GEMINI_CONFIG = {
  temperature: 0.7,
  maxOutputTokens: 1024,
};

function extractGeminiText(candidate) {
  return candidate?.content?.parts
    ?.map(part => part?.text || "")
    .join("")
    .trim() || "";
}

function isCompleteQuestion(text) {
  return /[?.!]$/.test(text.trim());
}

function hasGreeting(text) {
  return /\b(hello|hi|welcome|good\s+(morning|afternoon|evening)|great\s+to\s+meet)\b/i.test(text);
}

function ensureOpeningGreeting(text, candidateName) {
  const trimmed = (text || "").trim();
  if (!trimmed) return trimmed;
  if (hasGreeting(trimmed)) return trimmed;

  const name = candidateName?.trim() ? `, ${candidateName.trim()}` : "";
  return `Welcome${name}. Thank you for joining today. ${trimmed}`;
}

function getLastMessageByRole(list, role) {
  for (let i = list.length - 1; i >= 0; i--) {
    if (list[i]?.role === role) return list[i];
  }
  return null;
}

function isLowEffortAnswer(text) {
  const trimmed = (text || "").trim();
  if (trimmed.length < 8) return true;
  const words = trimmed.split(/\s+/).filter(Boolean);
  if (words.length < 2) return true;
  return /^(ok|okay|yes|no|idk|i\s*don'?t\s*know|hmm|hmmm|na|nah|yep|nope)$/i.test(trimmed);
}

function evaluateAnswerLocally({ question, answer, role, phase, responseTime }) {
  const answerText = (answer || "").trim();
  const questionText = (question || "").toLowerCase();
  const roleText = (role || "").toLowerCase();
  const phaseText = (phase || "").toLowerCase();
  const words = answerText.split(/\s+/).filter(Boolean);
  const wordCount = words.length;
  const sentenceCount = Math.max(1, answerText.split(/[.!?]+/).filter(Boolean).length);

  const relevanceHintWords = new Set(
    `${questionText} ${roleText} ${phaseText}`
      .split(/[^a-z0-9+#]+/)
      .filter(w => w.length >= 4)
  );
  const matched = words.filter(w => relevanceHintWords.has(w.toLowerCase())).length;

  const correctness = Math.max(2, Math.min(10, Math.round(3 + Math.min(wordCount, 90) / 12)));
  const relevance = Math.max(2, Math.min(10, Math.round(3 + matched * 1.2)));
  const clarity = Math.max(2, Math.min(10, Math.round(4 + Math.min(sentenceCount, 6))));
  const depth = Math.max(1, Math.min(10, Math.round(2 + Math.min(wordCount, 120) / 18)));
  const avg = (correctness + relevance + clarity + depth) / 4;

  return {
    quality: avg >= 7 ? "strong" : avg >= 4 ? "average" : "weak",
    scores: { correctness, relevance, clarity, depth },
    feedback: avg >= 7
      ? "Strong answer with good structure and relevant detail."
      : avg >= 4
      ? "Reasonable answer; add concrete examples and deeper technical detail."
      : "Answer was brief or generic; provide a more specific, structured response.",
    questionRelevance: Math.max(3, Math.min(10, relevance)),
    responseTime,
  };
}

function normalizeEvalResult(result, responseTime) {
  if (!result || typeof result !== "object") return null;

  const scores = result.scores || {};
  const normalized = {
    quality: ["strong", "average", "weak"].includes(result.quality) ? result.quality : "average",
    scores: {
      correctness: Number(scores.correctness),
      relevance: Number(scores.relevance),
      clarity: Number(scores.clarity),
      depth: Number(scores.depth),
    },
    feedback: typeof result.feedback === "string" ? result.feedback.trim() : "",
    questionRelevance: Number(result.questionRelevance),
    responseTime,
  };

  const allValid = Object.values(normalized.scores).every(value => Number.isFinite(value) && value >= 0 && value <= 10)
    && Number.isFinite(normalized.questionRelevance)
    && normalized.questionRelevance >= 0
    && normalized.questionRelevance <= 10;

  return allValid ? normalized : null;
}

function clamp100(value, fallback = 50) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(100, Math.round(n)));
}

function verdictFromScore(score) {
  if (score >= 85) return "Strongly Recommended";
  if (score >= 70) return "Recommended";
  if (score >= 50) return "Borderline";
  return "Not Recommended";
}

// ─── API ──────────────────────────────────────────────────────────────────────
async function callGemini(prompt, options = {}) {
  if (!GEMINI_API_KEY || !GEMINI_URL) {
    throw new Error("Missing Gemini API key. Set VITE_GEMINI_API_KEY in Mesra/.env or interview-sim/.env and restart Vite.");
  }

  const { retries = 2, ...generationOverrides } = options;
  const generationConfig = {
    ...DEFAULT_GEMINI_CONFIG,
    ...generationOverrides,
  };

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(GEMINI_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.error?.message || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const candidate = data.candidates?.[0];
      const text = extractGeminiText(candidate);
      if (candidate?.finishReason === "MAX_TOKENS") {
        throw new Error("Gemini response was truncated before completion.");
      }
      if (!text) throw new Error("Empty response from Gemini");
      return text;
    } catch (e) {
      if (attempt === retries) throw e;
      await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
    }
  }
}

function safeParseJSON(text) {
  try {
    return JSON.parse(text.replace(/```json\s*/gi, "").replace(/```\s*/g, "").trim());
  } catch {
    const match = text.match(/\{[\s\S]*\}/);
    if (match) { try { return JSON.parse(match[0]); } catch { return null; } }
    return null;
  }
}

// ─── PROMPTS ──────────────────────────────────────────────────────────────────
function buildQuestionPrompt({ phase, role, jd, experience, resumeText, history, lastAnswerQuality, candidateName }) {
  const transcript = history.map(m => `${m.role === "ai" ? "Interviewer" : "Candidate"}: ${m.text}`).join("\n");
  const isOpeningTurn = history.length === 0;
  const adaptiveNote = lastAnswerQuality === "weak"
    ? "The candidate's last answer was incomplete or shallow. Ask a probing follow-up or a simpler clarifying question."
    : lastAnswerQuality === "strong"
    ? "The candidate answered very well. Increase difficulty — probe an edge case or add complexity."
    : "Continue naturally to the next question for this phase.";
  return `You are an experienced senior enterprise interviewer conducting a structured job interview.

CANDIDATE PROFILE:
- Candidate name: ${candidateName || "Not provided"}
- Role applied for: ${role}
- Experience level: ${experience} years
- JD / Key requirements: ${jd || "Not provided — infer from role."}
- Resume Highlights: ${resumeText ? resumeText.slice(0, 1200) : "Not provided — infer from role and experience level."}

CURRENT PHASE: ${phase}
ADAPTIVE INSTRUCTION: ${lastAnswerQuality ? adaptiveNote : "This is the opening question. Begin warmly and professionally."}

FULL TRANSCRIPT SO FAR:
${transcript || "(Interview just started)"}

PHASE GUIDELINES:
- Introduction: Background, motivation, self-introduction — conversational, warm
- Technical: Deep domain-specific knowledge for the role
- Problem Solving: Real-world scenario — evaluate structured thinking and approach
- Managerial: Leadership, teamwork, conflict resolution, stakeholder communication
- Closing: Candidate's questions, career goals, final impressions

STRICT RULES:
1. Output ONLY ONE interview question relevant to the role and JD. The question must evaluate real-world skills required for the role. Do not include commentary, explanation, or evaluation.
2. ONE question only. Keep it concise and professional.
3. Tailor difficulty to ${experience} years of experience.
4. Never repeat a question already asked in the transcript.
5. The question must directly assess competencies expected for this specific role.
6. ${isOpeningTurn
  ? "Because this is the first turn, start with ONE short, warm greeting to the candidate, then ask the first interview question in the same message."
  : "Do not add extra greeting text; continue with only the next interview question."}`;
}

function buildAnswerEvalPrompt({ question, answer, role, phase, responseTime }) {
  return `You are an expert interview evaluator. Assess this candidate response objectively.

Role: ${role}
Interview Phase: ${phase}
Question Asked: ${question}
Candidate's Answer: ${answer}
Candidate response time: ${responseTime ? `${responseTime} seconds` : "not recorded"}

Evaluate and return ONLY valid JSON (no markdown fences, no explanation outside JSON):
{
  "quality": "strong",
  "scores": {
    "correctness": 8,
    "relevance": 7,
    "clarity": 9,
    "depth": 6
  },
  "feedback": "One concise sentence stating what was strong or what was missing.",
  "questionRelevance": 9
}

Scoring guide (all fields 0–10):
- quality: "strong" (7–10 avg), "average" (4–6 avg), "weak" (0–3 avg)
- correctness: factual/technical accuracy
- relevance: how well the answer addresses the question
- clarity: communication quality, structure
- depth: level of insight and detail
- questionRelevance: how relevant this question was to the stated role and JD`;
}

async function generateInterviewQuestion(args) {
  const primary = await callGemini(buildQuestionPrompt(args), {
    temperature: 0.55,
    maxOutputTokens: 256,
    thinkingConfig: { thinkingBudget: 0 },
  });

  if (isCompleteQuestion(primary)) return primary;

  const repaired = await callGemini(
    `Rewrite the following as one complete, concise interview question. Return only the finished question.\n\n${primary}`,
    {
      temperature: 0.2,
      maxOutputTokens: 128,
      thinkingConfig: { thinkingBudget: 0 },
      retries: 1,
    }
  );

  return repaired;
}

async function generateOpeningQuestion(args) {
  const opening = await generateInterviewQuestion(args);
  return ensureOpeningGreeting(opening, args.candidateName);
}

async function evaluateAnswer(args) {
  const raw = await callGemini(buildAnswerEvalPrompt(args), {
    temperature: 0.1,
    maxOutputTokens: 512,
    responseMimeType: "application/json",
    thinkingConfig: { thinkingBudget: 0 },
  });
  return normalizeEvalResult(safeParseJSON(raw), args.responseTime);
}

async function generateScorecard(args) {
  const raw = await callGemini(buildFinalScorecardPrompt(args), {
    temperature: 0.2,
    maxOutputTokens: 3072,
    responseMimeType: "application/json",
    thinkingConfig: { thinkingBudget: 0 },
    retries: 3,
  });
  return normalizeScorecardResult(safeParseJSON(raw), args);
}

function calcBaseScore(perAnswerEvals) {
  if (!perAnswerEvals.length) return 50;
  const avg = perAnswerEvals.reduce((sum, e) => {
    const s = e.scores || {};
    return sum + ((s.correctness || 0) + (s.relevance || 0) + (s.clarity || 0) + (s.depth || 0)) / 4;
  }, 0) / perAnswerEvals.length;
  return Math.round(avg * 10); // convert 0-10 → 0-100
}

function getResponseTimeStats(perAnswerEvals) {
  const times = perAnswerEvals
    .map(e => Number(e?.responseTime))
    .filter(t => Number.isFinite(t) && t > 0);

  if (!times.length) return null;

  const total = times.reduce((sum, t) => sum + t, 0);
  return {
    average: Math.round(total / times.length),
    fastest: Math.min(...times),
    slowest: Math.max(...times),
    samples: times.length,
  };
}

function getSkillCoverage({ messages, role, jd }) {
  const aiQuestions = messages
    .filter(m => m.role === "ai")
    .map(m => `${m.phase || ""} ${m.text || ""}`)
    .join("\n")
    .toLowerCase();
  const context = `${role || ""} ${jd || ""}`.toLowerCase();
  const corpus = `${aiQuestions}\n${context}`;

  const keywordRules = [
    { skill: "System Design", regex: /(system\s*design|architecture|scalab|distributed|microservice|component\s*design)/i },
    { skill: "API Design", regex: /(api|rest|graphql|endpoint|contract|versioning|http)/i },
    { skill: "Debugging", regex: /(debug|bug|troubleshoot|issue|root\s*cause|fix)/i },
    { skill: "Database Design", regex: /(database|sql|nosql|schema|index|query|normalization)/i },
    { skill: "Team Communication", regex: /(team|stakeholder|communication|collaborat|conflict|cross-functional)/i },
    { skill: "Leadership", regex: /(lead|mentor|ownership|decision|strategy|manage)/i },
    { skill: "Problem Solving", regex: /(problem|approach|trade-?off|optimi|analysis|scenario)/i },
    { skill: "Testing & Quality", regex: /(test|qa|unit\s*test|integration|quality|reliability)/i },
  ];

  const skills = keywordRules
    .filter(rule => rule.regex.test(corpus))
    .map(rule => rule.skill);

  // Add phase-derived coverage if keywords were sparse
  const phasesAsked = new Set(messages.filter(m => m.role === "ai").map(m => m.phase));
  if (phasesAsked.has("Technical") && !skills.includes("Problem Solving")) skills.push("Problem Solving");
  if (phasesAsked.has("Managerial") && !skills.includes("Team Communication")) skills.push("Team Communication");
  if (phasesAsked.has("Problem Solving") && !skills.includes("Debugging")) skills.push("Debugging");

  if (!skills.length) {
    return ["Problem Solving", "Role Knowledge", "Communication"];
  }

  return Array.from(new Set(skills)).slice(0, 8);
}

function buildSessionAnalytics({ messages, perAnswerEvals, role, jd }) {
  return {
    responseTime: getResponseTimeStats(perAnswerEvals),
    skillsEvaluated: getSkillCoverage({ messages, role, jd }),
  };
}

function generateLocalScorecard({ role, perAnswerEvals, tabSwitchCount }) {
  const base = buildFallbackScorecard({ role, perAnswerEvals, tabSwitchCount });
  base.candidateScore.summary = `Generated from local interview analytics for the ${role} role. This mode minimizes API usage while preserving structured scoring.`;
  return base;
}

function normalizeScorecardResult(result, { role, perAnswerEvals }) {
  if (!result || typeof result !== "object") return null;

  const baseScore = calcBaseScore(perAnswerEvals);
  const cs = result.candidateScore || {};
  const is_ = result.interviewerScore || {};
  const hr = result.hiringRecommendation || {};

  const normalized = {
    candidateScore: {
      overall: clamp100(cs.overall, baseScore),
      technical: clamp100(cs.technical, baseScore),
      communication: clamp100(cs.communication, baseScore),
      problemSolving: clamp100(cs.problemSolving, baseScore),
      roleRelevance: clamp100(cs.roleRelevance, baseScore),
      summary: typeof cs.summary === "string" && cs.summary.trim()
        ? cs.summary.trim()
        : `Performance was assessed from interview metrics for the ${role} role.`,
    },
    interviewerScore: {
      overall: clamp100(is_.overall, 78),
      questionQuality: clamp100(is_.questionQuality, 78),
      progressionLogic: clamp100(is_.progressionLogic, 76),
      coverageBreadth: clamp100(is_.coverageBreadth, 78),
      followUpIntelligence: clamp100(is_.followUpIntelligence, 76),
      summary: typeof is_.summary === "string" && is_.summary.trim()
        ? is_.summary.trim()
        : "Interviewer effectiveness was estimated from progression and relevance telemetry.",
    },
    strengths: Array.isArray(result.strengths) ? result.strengths.filter(Boolean).slice(0, 5) : [],
    improvements: Array.isArray(result.improvements) ? result.improvements.filter(Boolean).slice(0, 5) : [],
    hiringRecommendation: {
      verdict: ["Strongly Recommended", "Recommended", "Borderline", "Not Recommended"].includes(hr.verdict)
        ? hr.verdict
        : verdictFromScore(clamp100(hr.suitabilityScore, baseScore)),
      reason: typeof hr.reason === "string" && hr.reason.trim()
        ? hr.reason.trim()
        : `Recommendation is based on technical quality, communication, and role fit for ${role}.`,
      suitabilityScore: clamp100(hr.suitabilityScore, baseScore),
    },
  };

  if (!normalized.strengths.length) {
    normalized.strengths = [
      "Completed multiple interview phases with measurable responses.",
      "Maintained baseline relevance to role-focused prompts.",
    ];
  }
  if (!normalized.improvements.length) {
    normalized.improvements = [
      "Add concrete implementation examples to increase depth.",
      "Expand answers with clearer trade-off reasoning.",
    ];
  }

  return normalized;
}

function buildFallbackScorecard({ role, perAnswerEvals, tabSwitchCount }) {
  const count = perAnswerEvals.length || 1;
  const avg = (key) => perAnswerEvals.reduce((sum, e) => sum + (e.scores?.[key] || 0), 0) / count;
  const avgQuestionRel = perAnswerEvals.reduce((sum, e) => sum + (e.questionRelevance || 0), 0) / count;
  const baseScore = calcBaseScore(perAnswerEvals);

  const technical = clamp100(avg("correctness") * 10, baseScore);
  const communication = clamp100(avg("clarity") * 10, baseScore);
  const problemSolving = clamp100(avg("depth") * 10, baseScore);
  const roleRelevance = clamp100(avg("relevance") * 10, baseScore);
  const suitabilityScore = clamp100(Math.round((technical + communication + problemSolving + roleRelevance) / 4), baseScore);

  const questionQuality = clamp100(avgQuestionRel * 10, 75);
  const progressionLogic = clamp100(65 + Math.min(30, perAnswerEvals.length * 3), 75);
  const coverageBreadth = clamp100(60 + Math.min(35, perAnswerEvals.length * 3), 75);
  const followUpIntelligence = clamp100(62 + Math.min(30, perAnswerEvals.length * 2), 75);
  const interviewerOverall = clamp100(Math.round((questionQuality + progressionLogic + coverageBreadth + followUpIntelligence) / 4), 75);

  return {
    candidateScore: {
      overall: suitabilityScore,
      technical,
      communication,
      problemSolving,
      roleRelevance,
      summary: `This report used computed metrics because the AI scorecard output was invalid. Candidate performance for ${role} appears ${suitabilityScore >= 70 ? "solid" : suitabilityScore >= 50 ? "mixed" : "below expectation"} overall.`,
    },
    interviewerScore: {
      overall: interviewerOverall,
      questionQuality,
      progressionLogic,
      coverageBreadth,
      followUpIntelligence,
      summary: "Interviewer metrics were estimated from progression and question relevance signals in this session.",
    },
    strengths: [
      "Completed multiple interview phases with measurable answer quality.",
      "Maintained role relevance across generated questions.",
    ],
    improvements: [
      "Provide deeper technical detail and explicit trade-offs in answers.",
      "Use structured examples to improve clarity and depth scores.",
    ],
    hiringRecommendation: {
      verdict: verdictFromScore(suitabilityScore),
      reason: `Recommendation is based on computed scoring metrics${tabSwitchCount > 0 ? " and session integrity signals" : ""}.`,
      suitabilityScore,
    },
  };
}

function buildFinalScorecardPrompt({ role, jd, experience, resumeText, history, perAnswerEvals, tabSwitchCount }) {
  const transcript = history
    .slice(-16)
    .map(m => `${m.role === "ai" ? "Interviewer" : "Candidate"}: ${String(m.text || "").slice(0, 280)}`)
    .join("\n");
  const evalSummary = perAnswerEvals.map((e, i) =>
    `Q${i + 1}: quality=${e.quality || "?"}, correctness=${e.scores?.correctness ?? "?"}, relevance=${e.scores?.relevance ?? "?"}, clarity=${e.scores?.clarity ?? "?"}, depth=${e.scores?.depth ?? "?"}, responseTime=${e.responseTime ?? "?"}s, questionRelevance=${e.questionRelevance ?? "?"}`
  ).join("\n");
  const baseScore = calcBaseScore(perAnswerEvals);
  return `You are an enterprise HR analytics engine. Generate a comprehensive, unbiased interview evaluation report.

SESSION DETAILS:
- Role: ${role}
- Experience Level: ${experience}
- JD: ${jd || "Not provided"}
- Resume: ${resumeText ? resumeText.slice(0, 800) : "Not provided"}
- Integrity: ${tabSwitchCount > 0 ? `⚠ Candidate switched tabs ${tabSwitchCount} time(s) during interview` : "No integrity issues detected"}

CODE-CALCULATED BASE SCORE: ${baseScore}/100
Use this as a reference anchor when generating candidateScore.overall — your overall score should be within ±10 of this value unless there is strong qualitative reason to deviate.

FULL TRANSCRIPT:
${transcript || "Transcript unavailable"}

PER-QUESTION EVALUATION DATA (includes response time in seconds):
${evalSummary}

Return ONLY valid JSON (absolutely no markdown, no text outside the JSON object):
{
  "candidateScore": {
    "overall": 72,
    "technical": 68,
    "communication": 80,
    "problemSolving": 70,
    "roleRelevance": 74,
    "summary": "2–3 balanced sentences assessing the candidate's performance."
  },
  "interviewerScore": {
    "overall": 85,
    "questionQuality": 88,
    "progressionLogic": 82,
    "coverageBreadth": 86,
    "followUpIntelligence": 84,
    "summary": "2 sentences assessing the quality and structure of the interview itself."
  },
  "strengths": [
    "Specific strength 1",
    "Specific strength 2",
    "Specific strength 3"
  ],
  "improvements": [
    "Specific improvement area 1",
    "Specific improvement area 2"
  ],
  "hiringRecommendation": {
    "verdict": "Recommended",
    "reason": "1–2 sentences justifying the recommendation based on role fit and interview performance.",
    "suitabilityScore": 74
  }
}

All numeric scores must be integers 0–100. verdict must be one of: "Strongly Recommended", "Recommended", "Borderline", "Not Recommended".`;
}

// ─── RESUME TEXT EXTRACTOR ───────────────────────────────────────────────────
async function extractResumeText(file) {
  // For .txt files read directly; for .pdf/.docx extract as text via FileReader
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result || "");
    reader.onerror = () => resolve("");
    // Read as text — works perfectly for .txt; gives raw text for .pdf too (good enough for prompts)
    reader.readAsText(file);
  });
}

// ─── CSS ──────────────────────────────────────────────────────────────────────
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080810;--surface:#10101a;--surface2:#181825;--surface3:#202030;
  --border:#252535;--border2:#303045;
  --accent:#7c6aff;--accent2:#ff6a9c;--accent3:#6affd4;--accent4:#ffb86a;
  --text:#eeeef5;--muted:#55556e;--muted2:#80809a;
  --success:#4ade80;--warning:#fbbf24;--danger:#f87171;--info:#60a5fa;
  --radius:10px;--radius-lg:16px;
}
:root[data-theme='light']{
  --bg:#f3f5fb;--surface:#ffffff;--surface2:#f7f8fc;--surface3:#eef1f9;
  --border:#d9deeb;--border2:#c8d0e0;
  --accent:#4f46e5;--accent2:#db2777;--accent3:#0ea5a4;--accent4:#f59e0b;
  --text:#1f2937;--muted:#6b7280;--muted2:#4b5563;
  --success:#16a34a;--warning:#ca8a04;--danger:#dc2626;--info:#2563eb;
}
body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;overflow-x:hidden}
.app{min-height:100vh;position:relative}
.app::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:radial-gradient(ellipse 80% 50% at 20% 10%,rgba(124,106,255,.07) 0%,transparent 60%),
             radial-gradient(ellipse 60% 40% at 80% 90%,rgba(255,106,156,.05) 0%,transparent 60%)}
.z1{position:relative;z-index:1}
.mono{font-family:'DM Mono',monospace}

/* Tags */
.tag{display:inline-flex;align-items:center;padding:.18rem .5rem;border-radius:5px;font-size:.62rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}
.ta{background:rgba(124,106,255,.12);color:var(--accent);border:1px solid rgba(124,106,255,.2)}
.ts{background:rgba(74,222,128,.1);color:var(--success);border:1px solid rgba(74,222,128,.2)}
.tw{background:rgba(251,191,36,.1);color:var(--warning);border:1px solid rgba(251,191,36,.2)}
.td{background:rgba(248,113,113,.1);color:var(--danger);border:1px solid rgba(248,113,113,.2)}
.tm{background:rgba(96,96,122,.12);color:var(--muted2);border:1px solid var(--border)}

/* Animations */
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-5px)}}

/* Buttons */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:.45rem;padding:.72rem 1.4rem;border-radius:var(--radius);font-family:'Syne',sans-serif;font-weight:700;font-size:.85rem;cursor:pointer;border:none;transition:all .2s;letter-spacing:.01em}
.bp{background:var(--accent);color:#fff}.bp:hover:not(:disabled){background:#9580ff;transform:translateY(-1px);box-shadow:0 6px 20px rgba(124,106,255,.35)}
.bs{background:var(--surface2);color:var(--text);border:1px solid var(--border2)}.bs:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
.bg{background:transparent;color:var(--muted2);border:1px solid var(--border)}.bg:hover:not(:disabled){color:var(--text);border-color:var(--border2)}
.bda{background:rgba(248,113,113,.08);color:var(--danger);border:1px solid rgba(248,113,113,.2)}.bda:hover:not(:disabled){background:rgba(248,113,113,.15)}
.bsu{background:rgba(74,222,128,.08);color:var(--success);border:1px solid rgba(74,222,128,.2)}.bsu:hover:not(:disabled){background:rgba(74,222,128,.15)}
.btn:disabled{opacity:.35;cursor:not-allowed}
.bsm{padding:.48rem .95rem;font-size:.78rem}
.bfw{width:100%}

/* Forms */
.field{margin-bottom:1.05rem}
.field label{display:block;font-size:.68rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--muted2);margin-bottom:.42rem}
.field input,.field select,.field textarea{width:100%;background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:.68rem .9rem;border-radius:var(--radius);font-family:'Syne',sans-serif;font-size:.875rem;outline:none;transition:border-color .2s,box-shadow .2s}
.field input:focus,.field select:focus,.field textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(124,106,255,.1)}
.field select option{background:var(--surface2)}
.field textarea{resize:vertical;min-height:72px;line-height:1.5}
.frow{display:grid;grid-template-columns:1fr 1fr;gap:.9rem}

/* Error */
.err-box{background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.22);border-radius:var(--radius);padding:.65rem .9rem;font-size:.8rem;color:var(--danger);display:flex;align-items:flex-start;gap:.55rem;animation:fadeUp .2s ease;line-height:1.5}
.err-box svg{flex-shrink:0;margin-top:1px}

/* ════════════════════ LANDING ════════════════════ */
.landing{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh;padding:2rem;text-align:center}
.hero-badge{display:inline-flex;align-items:center;gap:.5rem;background:rgba(124,106,255,.07);border:1px solid rgba(124,106,255,.22);color:var(--accent);padding:.28rem .8rem;border-radius:999px;font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:1.75rem}
.hero-badge::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--accent);animation:pulse 2s infinite}
h1.htitle{font-size:clamp(2.2rem,5.5vw,4.5rem);font-weight:800;line-height:1.04;letter-spacing:-.03em;margin-bottom:1.1rem}
.grad{background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hdesc{font-size:.95rem;color:var(--muted2);max-width:460px;line-height:1.7;margin-bottom:2.5rem;font-family:'DM Mono',monospace}
.scard{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:2.1rem;width:100%;max-width:490px;text-align:left}
.scard-title{font-size:.95rem;font-weight:700;margin-bottom:1.35rem;display:flex;align-items:center;gap:.5rem}
.scard-title::before{content:'◈';color:var(--accent)}
.mtoggle{display:grid;grid-template-columns:1fr 1fr;gap:.4rem;background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:.28rem;margin-bottom:1.35rem}
.mbtn{padding:.5rem;border-radius:7px;text-align:center;font-size:.78rem;font-weight:700;cursor:pointer;transition:all .2s;border:none;font-family:'Syne',sans-serif;color:var(--muted2);background:transparent}
.mbtn.on{background:var(--accent);color:#fff;box-shadow:0 2px 8px rgba(124,106,255,.3)}
.hfeats{display:flex;gap:1.75rem;color:var(--muted);font-size:.68rem;font-family:'DM Mono',monospace;margin-top:1.75rem;flex-wrap:wrap;justify-content:center}
.resume-drop{border:1.5px dashed var(--border2);border-radius:var(--radius);padding:.85rem 1rem;cursor:pointer;transition:border-color .2s,background .2s;text-align:center;font-size:.78rem;color:var(--muted2);background:var(--surface2)}
.resume-drop:hover{border-color:var(--accent);background:rgba(124,106,255,.04);color:var(--text)}
.resume-drop.has-file{border-color:var(--success);color:var(--success);background:rgba(74,222,128,.05)}
.resume-input{display:none}

/* ════════════════════ INTERVIEW ════════════════════ */
.iroot{display:grid;grid-template-columns:255px 1fr;min-height:100vh}
.iroot.with-panel{grid-template-columns:255px 1fr 265px}

.sidebar{background:var(--surface);border-right:1px solid var(--border);padding:1.3rem;display:flex;flex-direction:column;gap:1.3rem;position:sticky;top:0;height:100vh;overflow-y:auto}
.sidebar::-webkit-scrollbar{width:3px}
.sidebar::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.sbrand{font-size:.72rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);display:flex;align-items:center;gap:.4rem}
.sbrand::before{content:'◈'}
.ccrd{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:.85rem}
.ccrd .cn{font-weight:700;font-size:.88rem;margin-bottom:.18rem}
.ccrd .cr{font-size:.7rem;color:var(--accent);font-family:'DM Mono',monospace;margin-bottom:.35rem}
.ccrd .cx{font-size:.68rem;color:var(--muted2);font-family:'DM Mono',monospace}
.slbl{font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:.55rem}
.plist{display:flex;flex-direction:column;gap:.28rem}
.prow{display:flex;align-items:center;gap:.55rem;padding:.42rem .52rem;border-radius:7px;font-size:.76rem;transition:background .15s}
.prow.on{background:rgba(124,106,255,.1);color:var(--accent);font-weight:600}
.prow.dn{color:var(--success)}
.prow.nd{color:var(--muted)}
.pdot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.prow.on .pdot{background:var(--accent);box-shadow:0 0 6px var(--accent)}
.prow.dn .pdot{background:var(--success)}
.prow.nd .pdot{background:var(--border2)}
.qprog{font-family:'DM Mono',monospace;font-size:.7rem;color:var(--muted)}
.qprog b{color:var(--text);font-weight:600}
.leval{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:.8rem}
.leval-title{font-size:.62rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);margin-bottom:.65rem}
.mbar-row{display:flex;flex-direction:column;gap:.18rem;margin-bottom:.48rem}
.mbar-top{display:flex;justify-content:space-between;font-size:.66rem}
.mbar-top span:first-child{color:var(--muted2)}
.mbar-top span:last-child{font-family:'DM Mono',monospace}
.btr{height:3px;background:var(--border);border-radius:2px;overflow:hidden}
.bfill{height:100%;border-radius:2px;transition:width .8s ease}
.mindfb{font-size:.66rem;color:var(--muted2);font-style:italic;margin-top:.4rem;line-height:1.45}
.mind-mode{display:flex;align-items:center;gap:.45rem;padding:.5rem .7rem;border-radius:var(--radius);font-size:.7rem;font-weight:700}
.mind-mode.int-m{background:rgba(255,184,106,.07);border:1px solid rgba(255,184,106,.2);color:var(--accent4)}
.mind-mode.cand-m{background:rgba(96,165,250,.07);border:1px solid rgba(96,165,250,.2);color:var(--info)}

/* Main col */
.mcol{display:flex;flex-direction:column;height:100vh;overflow:hidden}
.tbar{display:flex;align-items:center;justify-content:space-between;padding:.85rem 1.6rem;border-bottom:1px solid var(--border);background:rgba(8,8,16,.88);backdrop-filter:blur(16px);position:sticky;top:0;z-index:10;gap:.75rem;flex-shrink:0}
.tbar-l{display:flex;align-items:center;gap:.7rem}
.tbar-title{font-size:.82rem;font-weight:700;letter-spacing:-.01em}
.tbar-r{display:flex;align-items:center;gap:.7rem}
.ldot{width:6px;height:6px;border-radius:50%;background:var(--success);animation:pulse 2s infinite;flex-shrink:0}
.smtoggle{display:flex;gap:.3rem;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:.22rem}
.smbtn{padding:.38rem .8rem;border-radius:6px;font-size:.72rem;font-weight:700;cursor:pointer;border:none;font-family:'Syne',sans-serif;color:var(--muted2);background:transparent;transition:all .2s}
.smbtn.on{background:var(--accent);color:#fff}
.theme-btn{padding:.38rem .72rem;border-radius:6px;font-size:.7rem;font-weight:700;cursor:pointer;border:1px solid var(--border2);font-family:'Syne',sans-serif;color:var(--muted2);background:var(--surface2);transition:all .2s}
.theme-btn:hover{border-color:var(--accent);color:var(--accent)}

/* Chat */
.cfeed{flex:1;overflow-y:auto;padding:1.5rem 1.6rem;display:flex;flex-direction:column;gap:1.1rem}
.cfeed::-webkit-scrollbar{width:4px}
.cfeed::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.msg{display:flex;gap:.8rem;animation:fadeUp .22s ease}
.msg.ai{align-self:flex-start;max-width:78%}
.msg.user{align-self:flex-end;flex-direction:row-reverse;max-width:78%}
.mav{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:800;flex-shrink:0}
.msg.ai .mav{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}
.msg.user .mav{background:var(--surface2);border:1px solid var(--border2);color:var(--muted2)}
.mbody{display:flex;flex-direction:column;gap:.3rem}
.mphase{display:flex;align-items:center;gap:.35rem}
.mbub{background:var(--surface);border:1px solid var(--border);border-radius:11px;padding:.82rem 1rem;font-size:.862rem;line-height:1.65}
.msg.user .mbub{background:rgba(124,106,255,.07);border-color:rgba(124,106,255,.18)}
.mtime{font-size:.62rem;color:var(--muted);font-family:'DM Mono',monospace}
.evalpill{display:flex;align-items:flex-start;flex-wrap:wrap;gap:.5rem;background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:.48rem .8rem;font-size:.7rem;margin-top:.1rem}
.epdims{display:flex;gap:.45rem;flex-wrap:wrap;align-items:center}
.edim{display:flex;align-items:center;gap:.22rem;font-family:'DM Mono',monospace;font-size:.68rem}
.edim .dn{color:var(--muted2)}
.edim .dv{font-weight:600}
.epfb{font-size:.68rem;color:var(--muted2);font-style:italic;border-left:2px solid var(--border2);padding-left:.55rem;width:100%;line-height:1.45}
.tdots{display:flex;gap:3px;padding:.65rem .85rem}
.tdots span{width:5px;height:5px;border-radius:50%;background:var(--muted);animation:bounce 1.1s infinite}
.tdots span:nth-child(2){animation-delay:.18s}
.tdots span:nth-child(3){animation-delay:.36s}
.idock{padding:.9rem 1.6rem;border-top:1px solid var(--border);background:var(--surface);display:flex;gap:.6rem;align-items:flex-end;flex-shrink:0}
.idock textarea{flex:1;background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:.68rem .88rem;border-radius:var(--radius);font-family:'Syne',sans-serif;font-size:.862rem;outline:none;resize:none;min-height:42px;max-height:120px;line-height:1.5;transition:border-color .2s}
.idock textarea:focus{border-color:var(--accent)}
.idock textarea:disabled{opacity:.45;cursor:not-allowed}
.sbtn{width:42px;height:42px;border-radius:var(--radius);background:var(--accent);border:none;color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .2s;flex-shrink:0}
.sbtn:hover:not(:disabled){background:#9580ff;transform:translateY(-1px)}
.sbtn:disabled{opacity:.3;cursor:not-allowed;transform:none}
.hint{font-size:.64rem;color:var(--muted);font-family:'DM Mono',monospace;padding:.3rem 1.6rem .7rem;flex-shrink:0}

/* Interviewer panel */
.ipanel{background:var(--surface);border-left:1px solid var(--border);padding:1.3rem;display:flex;flex-direction:column;gap:1.15rem;position:sticky;top:0;height:100vh;overflow-y:auto}
.ipanel::-webkit-scrollbar{width:3px}
.ipanel::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.iphdr{font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.iqbox{background:var(--surface2);border:1px solid var(--border2);border-radius:var(--radius);padding:.8rem;font-size:.8rem;line-height:1.6;color:var(--text)}
.inotes textarea{width:100%;background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:.62rem .82rem;border-radius:var(--radius);font-family:'DM Mono',monospace;font-size:.72rem;outline:none;resize:vertical;min-height:75px;transition:border-color .2s}
.inotes textarea:focus{border-color:var(--accent)}
.ahist{display:flex;flex-direction:column;gap:.55rem}
.ahrow{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:.6rem .8rem}
.ahq{font-size:.68rem;color:var(--muted2);margin-bottom:.28rem;line-height:1.4}
.ahscores{display:flex;gap:.35rem;flex-wrap:wrap}

/* ════════════════════ SCORECARD ════════════════════ */
.scroot{min-height:100vh;padding:3rem 2rem 2rem;display:flex;flex-direction:column;align-items:center}
.schdr{text-align:center;margin-bottom:2.75rem}
.schdr h1{font-size:clamp(1.9rem,4.5vw,3.2rem);font-weight:800;letter-spacing:-.03em;margin-bottom:.45rem}
.schdr p{font-family:'DM Mono',monospace;font-size:.78rem;color:var(--muted2)}
.scgrid{display:grid;grid-template-columns:1fr 1fr;gap:1.35rem;width:100%;max-width:900px;margin-bottom:1.35rem}
.sccard{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.6rem}
.scctitle{font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:1.3rem}
.scbig{font-size:3.5rem;font-weight:800;letter-spacing:-.04em;line-height:1;margin-bottom:.28rem}
.scbig.g{color:var(--success)}.scbig.y{color:var(--warning)}.scbig.r{color:var(--danger)}.scbig.p{color:var(--accent)}
.scsum{font-size:.76rem;color:var(--muted2);line-height:1.55;margin-bottom:1.2rem}
.scbars{display:flex;flex-direction:column;gap:.65rem}
.sbarrow{display:flex;flex-direction:column;gap:.2rem}
.sbarlbl{display:flex;justify-content:space-between;font-size:.68rem}
.sbarlbl span:first-child{color:var(--muted2)}
.sbarlbl span:last-child{font-family:'DM Mono',monospace;font-weight:600}
.scins{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.6rem;width:100%;max-width:900px;margin-bottom:1.35rem}
.inslist{display:flex;flex-direction:column;gap:.45rem}
.insrow{display:flex;gap:.6rem;align-items:flex-start;font-size:.8rem;line-height:1.55;padding:.6rem .8rem;background:var(--surface2);border-radius:8px}
.insicon{flex-shrink:0;font-size:.85rem}
.scperq{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:1.6rem;width:100%;max-width:900px;margin-bottom:1.35rem;overflow-x:auto}
.ptbl{width:100%;border-collapse:collapse;font-size:.75rem}
.ptbl th{text-align:left;font-size:.6rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);padding:.38rem .55rem;border-bottom:1px solid var(--border);white-space:nowrap}
.ptbl td{padding:.52rem .55rem;border-bottom:1px solid var(--border);vertical-align:top;line-height:1.4}
.ptbl tr:last-child td{border-bottom:none}
.ptq{color:var(--muted2);max-width:240px}
.ptfb{color:var(--muted2);font-style:italic;max-width:180px;font-size:.7rem}
.scverdict{background:linear-gradient(135deg,rgba(124,106,255,.07),rgba(255,106,156,.07));border:1px solid rgba(124,106,255,.18);border-radius:var(--radius-lg);padding:1.85rem;width:100%;max-width:900px;text-align:center;margin-bottom:1.35rem}
.svlbl{font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:.45rem}
.svverd{font-size:1.5rem;font-weight:800;letter-spacing:-.02em;margin-bottom:.55rem}
.svreason{font-size:.78rem;color:var(--muted2);font-family:'DM Mono',monospace;line-height:1.55}
.svsuit{font-size:2.8rem;font-weight:800;letter-spacing:-.03em;margin-top:.9rem}
.scacts{display:flex;gap:.7rem;flex-wrap:wrap;justify-content:center;margin-bottom:3rem}

/* Loading */
.lscr{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh;gap:1.15rem}
.spin{width:34px;height:34px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite}
.lscr p{font-family:'DM Mono',monospace;font-size:.78rem;color:var(--muted2)}

@media(max-width:860px){
  .iroot,.iroot.with-panel{grid-template-columns:1fr}
  .sidebar,.ipanel{display:none}
  .scgrid{grid-template-columns:1fr}
  .frow{grid-template-columns:1fr}
}
`;

// ─── HELPERS ──────────────────────────────────────────────────────────────────
const scoreColor = s => s >= 75 ? "g" : s >= 50 ? "y" : "r";
const barColor = s => s >= 75 ? "#4ade80" : s >= 50 ? "#fbbf24" : "#f87171";

function QTag({ quality }) {
  if (!quality) return null;
  const map = { strong: ["ts", "Strong"], average: ["tw", "Average"], weak: ["td", "Needs Work"] };
  const [cls, lbl] = map[quality] || ["tm", quality];
  return <span className={`tag ${cls}`}>{lbl}</span>;
}

function SBar({ label, value, wide }) {
  const color = barColor(value);
  return wide ? (
    <div className="sbarrow">
      <div className="sbarlbl"><span>{label}</span><span>{value}/100</span></div>
      <div className="btr"><div className="bfill" style={{ width: `${value}%`, background: color }} /></div>
    </div>
  ) : (
    <div className="mbar-row">
      <div className="mbar-top"><span>{label}</span><span>{value}</span></div>
      <div className="btr"><div className="bfill" style={{ width: `${value}%`, background: color }} /></div>
    </div>
  );
}

function ErrBox({ msg, onDismiss }) {
  if (!msg) return null;
  return (
    <div className="err-box">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <span style={{ flex: 1 }}>{msg}</span>
      {onDismiss && <span style={{ cursor: "pointer" }} onClick={onDismiss}>✕</span>}
    </div>
  );
}

// ─── SCORECARD PAGE ───────────────────────────────────────────────────────────
function Scorecard({ scores, config, messages, perAnswerEvals, tabSwitchCount, onRestart, onExport, onExportPdf }) {
  if (!scores) return (
    <div className="scroot z1">
      <div style={{ maxWidth: 600, marginTop: "4rem", width: "100%" }}>
        <ErrBox msg="Could not generate scorecard — the AI returned an unexpected response. Please try a new session." />
      </div>
      <button className="btn bg" style={{ marginTop: "1.5rem" }} onClick={onRestart}>← Back</button>
    </div>
  );
  const cs = scores?.candidateScore || {
    overall: 0,
    technical: 0,
    communication: 0,
    problemSolving: 0,
    roleRelevance: 0,
    summary: "No candidate summary available.",
  };
  const is_ = scores?.interviewerScore || {
    overall: 0,
    questionQuality: 0,
    progressionLogic: 0,
    coverageBreadth: 0,
    followUpIntelligence: 0,
    summary: "No interviewer summary available.",
  };
  const strengths = Array.isArray(scores?.strengths) ? scores.strengths : [];
  const improvements = Array.isArray(scores?.improvements) ? scores.improvements : [];
  const hr = scores?.hiringRecommendation || { verdict: "—", reason: "", suitabilityScore: 0 };
  const aiMsgs = messages.filter(m => m.role === "ai");
  const suit = hr?.suitabilityScore ?? 0;
  const baseScore = calcBaseScore(perAnswerEvals);
  const analytics = buildSessionAnalytics({ messages, perAnswerEvals, role: config.role, jd: config.jd });
  const rt = analytics.responseTime;
  return (
    <div className="scroot z1">
      <div className="schdr">
        <div className="hero-badge" style={{ marginBottom: ".9rem" }}>Session Complete</div>
        <h1>Interview <span className="grad">Report</span></h1>
        <p>{config.candidateName} · {config.role} · {config.experience} yrs · {new Date().toLocaleDateString("en-IN", { day: "numeric", month: "long", year: "numeric" })}</p>
      </div>
      <div className="scgrid">
        <div className="sccard">
          <div className="scctitle">Candidate Performance</div>
          <div className={`scbig ${scoreColor(cs.overall)}`}>{cs.overall}</div>
          <div className="scsum">{cs.summary}</div>
          <div className="scbars">
            <SBar label="Technical Knowledge" value={cs.technical} wide />
            <SBar label="Communication" value={cs.communication} wide />
            <SBar label="Problem Solving" value={cs.problemSolving} wide />
            <SBar label="Role Relevance" value={cs.roleRelevance} wide />
          </div>
        </div>
        <div className="sccard">
          <div className="scctitle">Interviewer Effectiveness</div>
          <div className="scbig p">{is_.overall}</div>
          <div className="scsum">{is_.summary}</div>
          <div className="scbars">
            <SBar label="Question Quality" value={is_.questionQuality} wide />
            <SBar label="Progression Logic" value={is_.progressionLogic} wide />
            <SBar label="Coverage Breadth" value={is_.coverageBreadth} wide />
            <SBar label="Follow-up Intelligence" value={is_.followUpIntelligence} wide />
          </div>
        </div>
      </div>
      <div className="scins">
        <div className="scctitle" style={{ marginBottom: ".9rem" }}>Key Insights</div>
        <div className="inslist">
          {(strengths || []).map((s, i) => <div key={i} className="insrow"><span className="insicon">✦</span><span>{s}</span></div>)}
          {(improvements || []).map((s, i) => <div key={i} className="insrow"><span className="insicon">△</span><span>{s}</span></div>)}
        </div>
      </div>
      {perAnswerEvals.length > 0 && (
        <div className="scperq">
          <div className="scctitle" style={{ marginBottom: ".9rem" }}>Per-Question Breakdown</div>
          <table className="ptbl">
            <thead>
              <tr>
                <th>#</th><th>Question</th><th>Phase</th><th>Quality</th>
                <th>Correctness</th><th>Relevance</th><th>Clarity</th><th>Depth</th>
                <th>Q.Relevance</th><th>⏱ Time</th><th>Feedback</th>
              </tr>
            </thead>
            <tbody>
              {perAnswerEvals.map((e, i) => (
                <tr key={i}>
                  <td style={{ fontFamily: "'DM Mono',monospace", color: "var(--muted2)" }}>{i + 1}</td>
                  <td className="ptq">{aiMsgs[i]?.text?.slice(0, 55)}{(aiMsgs[i]?.text?.length || 0) > 55 ? "…" : ""}</td>
                  <td><span className="tag ta" style={{ fontSize: ".58rem" }}>{aiMsgs[i]?.phase || "—"}</span></td>
                  <td><QTag quality={e.quality} /></td>
                  {["correctness", "relevance", "clarity", "depth"].map(k => (
                    <td key={k} style={{ fontFamily: "'DM Mono',monospace", color: barColor((e.scores?.[k] ?? 0) * 10), fontWeight: 600 }}>{e.scores?.[k] ?? "—"}/10</td>
                  ))}
                  <td style={{ fontFamily: "'DM Mono',monospace", color: barColor((e.questionRelevance ?? 0) * 10), fontWeight: 600 }}>{e.questionRelevance ?? "—"}/10</td>
                  <td style={{ fontFamily: "'DM Mono',monospace", color: "var(--muted2)" }}>{e.responseTime != null ? `${e.responseTime}s` : "—"}</td>
                  <td className="ptfb">{e.feedback || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {tabSwitchCount > 0 && (
        <div style={{ background: "rgba(248,113,113,.07)", border: "1px solid rgba(248,113,113,.2)", borderRadius: "var(--radius-lg)", padding: "1.1rem 1.5rem", width: "100%", maxWidth: 900, marginBottom: "1.35rem", display: "flex", alignItems: "center", gap: ".75rem", fontSize: ".82rem" }}>
          <span style={{ fontSize: "1.1rem" }}>⚠</span>
          <div>
            <div style={{ fontWeight: 700, color: "var(--danger)", marginBottom: ".25rem" }}>Integrity Warning</div>
            <div style={{ color: "var(--muted2)", fontFamily: "'DM Mono',monospace", fontSize: ".75rem" }}>Candidate switched tabs {tabSwitchCount} time(s) during the interview session. This has been logged in the exported report.</div>
          </div>
        </div>
      )}
      <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "1.25rem 1.75rem", width: "100%", maxWidth: 900, marginBottom: "1.35rem", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem" }}>
        <div>
          <div style={{ fontSize: ".62rem", fontWeight: 700, letterSpacing: ".08em", textTransform: "uppercase", color: "var(--muted)", marginBottom: ".3rem" }}>Code-Calculated Base Score</div>
          <div style={{ fontSize: ".75rem", color: "var(--muted2)", fontFamily: "'DM Mono',monospace" }}>Computed from {perAnswerEvals.length} per-answer evaluations · Used to anchor AI scoring</div>
        </div>
        <div style={{ fontSize: "2.5rem", fontWeight: 800, letterSpacing: "-.03em", color: barColor(baseScore), flexShrink: 0 }}>{baseScore}<span style={{ fontSize: "1rem", color: "var(--muted2)" }}>/100</span></div>
      </div>
      <div className="scins">
        <div className="scctitle" style={{ marginBottom: ".9rem" }}>Interview Analytics</div>
        {rt ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: ".65rem", marginBottom: ".95rem" }}>
            <div className="insrow" style={{ justifyContent: "space-between", alignItems: "center" }}><span>Average Response Time</span><strong>{rt.average}s</strong></div>
            <div className="insrow" style={{ justifyContent: "space-between", alignItems: "center" }}><span>Fastest Answer</span><strong>{rt.fastest}s</strong></div>
            <div className="insrow" style={{ justifyContent: "space-between", alignItems: "center" }}><span>Slowest Answer</span><strong>{rt.slowest}s</strong></div>
          </div>
        ) : (
          <div className="insrow" style={{ marginBottom: ".95rem" }}>Response-time data is unavailable for this session.</div>
        )}
        <div style={{ fontSize: ".62rem", fontWeight: 700, letterSpacing: ".08em", textTransform: "uppercase", color: "var(--muted)", marginBottom: ".6rem" }}>Skills Evaluated</div>
        <div className="inslist">
          {analytics.skillsEvaluated.map((skill, i) => (
            <div key={i} className="insrow"><span className="insicon">✓</span><span>{skill}</span></div>
          ))}
        </div>
      </div>
      <div className="scverdict">
        <div className="svlbl">Enterprise Hiring Recommendation</div>
        <div className="svverd">{hr?.verdict ?? "—"}</div>
        <div className="svreason">{hr?.reason ?? ""}</div>
        <div className="svsuit" style={{ color: barColor(suit) }}>{suit}<span style={{ fontSize: "1.1rem", color: "var(--muted2)" }}>/100</span></div>
        <div style={{ fontSize: ".65rem", color: "var(--muted)", fontFamily: "'DM Mono',monospace", marginTop: ".25rem" }}>Overall Suitability Score</div>
      </div>
      <div className="scacts">
        <button className="btn bp" onClick={onExportPdf}>⬇ Download Report (PDF)</button>
        <button className="btn bsu" onClick={onExport}>⬇ Export Report (JSON)</button>
        <button className="btn bg" onClick={onRestart}>← New Interview</button>
      </div>
    </div>
  );
}

// ─── MAIN ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [screen, setScreen] = useState("landing");
  const [mode, setMode] = useState("candidate");
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem(THEME_STORAGE_KEY) || "dark";
    } catch {
      return "dark";
    }
  });
  const [config, setConfig] = useState({ candidateName: "", role: "", jd: "", experience: "2-5" });
  const [resumeText, setResumeText] = useState("");
  const [resumeFileName, setResumeFileName] = useState("");
  const [messages, setMessages] = useState([]);
  const [perAnswerEvals, setPerAnswerEvals] = useState([]);
  const [lastEval, setLastEval] = useState(null);
  const [interviewerNotes, setInterviewerNotes] = useState("");
  const [input, setInput] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [evalLoading, setEvalLoading] = useState(false);
  const [error, setError] = useState("");
  const [phaseIdx, setPhaseIdx] = useState(0);
  const [qInPhase, setQInPhase] = useState(0);
  const [totalAnswered, setTotalAnswered] = useState(0);
  const [scores, setScores] = useState(null);
  const [scorecardNotice, setScorecardNotice] = useState("");
  // #5 Response time tracking
  const [questionStartTime, setQuestionStartTime] = useState(null);
  // #9 Anti-cheat tab switching
  const [tabSwitchCount, setTabSwitchCount] = useState(0);
  const [tabWarning, setTabWarning] = useState("");
  const chatRef = useRef(null);
  const didRestoreRef = useRef(false);

  const totalQ = PHASES.length * QUESTIONS_PER_PHASE;

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Ignore storage failures in restricted environments.
    }
  }, [theme]);

  const toggleTheme = () => setTheme(prev => (prev === "dark" ? "light" : "dark"));

  const buildExportReportPayload = useCallback((forceScores) => {
    const exportScores = forceScores || scores;
    if (!exportScores) return null;
    const analytics = buildSessionAnalytics({ messages, perAnswerEvals, role: config.role, jd: config.jd });
    const baseScore = calcBaseScore(perAnswerEvals);

    return {
      meta: { generatedAt: new Date().toISOString(), platform: "InterviewSim" },
      candidate: { ...config, resumeProvided: !!resumeText },
      integrity: { tabSwitchCount, note: tabSwitchCount > 0 ? "Candidate switched tabs during interview" : "No issues" },
      analytics,
      baseScore,
      scores: exportScores,
      perQuestionEvaluation: perAnswerEvals,
      transcript: messages,
      interviewerNotes,
    };
  }, [scores, config, resumeText, tabSwitchCount, messages, perAnswerEvals, interviewerNotes]);

  // Restore in-progress session if the tab was refreshed/reloaded.
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(SESSION_STORAGE_KEY);
      if (!raw) return;
      const state = JSON.parse(raw);
      if (!state || typeof state !== "object") return;

      setScreen(state.screen || "landing");
      setMode(state.mode || "candidate");
      setConfig(state.config || { candidateName: "", role: "", jd: "", experience: "2-5" });
      setResumeText(state.resumeText || "");
      setResumeFileName(state.resumeFileName || "");
      setMessages(Array.isArray(state.messages) ? state.messages : []);
      setPerAnswerEvals(Array.isArray(state.perAnswerEvals) ? state.perAnswerEvals : []);
      setLastEval(state.lastEval || null);
      setInterviewerNotes(state.interviewerNotes || "");
      setInput(state.input || "");
      setError(state.error || "");
      setPhaseIdx(Number.isFinite(state.phaseIdx) ? state.phaseIdx : 0);
      setQInPhase(Number.isFinite(state.qInPhase) ? state.qInPhase : 0);
      setTotalAnswered(Number.isFinite(state.totalAnswered) ? state.totalAnswered : 0);
      setScores(state.scores || null);
      setScorecardNotice(state.scorecardNotice || "");
      setQuestionStartTime(Number.isFinite(state.questionStartTime) ? state.questionStartTime : null);
      setTabSwitchCount(Number.isFinite(state.tabSwitchCount) ? state.tabSwitchCount : 0);
      setTabWarning(state.tabWarning || "");
    } catch {
      // Ignore bad session cache and start clean.
    } finally {
      didRestoreRef.current = true;
    }
  }, []);

  // Persist session state so a tab refresh does not reset the interview.
  useEffect(() => {
    if (!didRestoreRef.current) return;
    try {
      const shouldPersist = screen !== "landing" || messages.length > 0 || perAnswerEvals.length > 0;
      if (!shouldPersist) {
        sessionStorage.removeItem(SESSION_STORAGE_KEY);
        return;
      }

      const snapshot = {
        screen,
        mode,
        config,
        resumeText,
        resumeFileName,
        messages,
        perAnswerEvals,
        lastEval,
        interviewerNotes,
        input,
        error,
        phaseIdx,
        qInPhase,
        totalAnswered,
        scores,
        scorecardNotice,
        questionStartTime,
        tabSwitchCount,
        tabWarning,
      };
      sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(snapshot));
    } catch {
      // Storage can fail in private mode; app still functions.
    }
  }, [
    screen,
    mode,
    config,
    resumeText,
    resumeFileName,
    messages,
    perAnswerEvals,
    lastEval,
    interviewerNotes,
    input,
    error,
    phaseIdx,
    qInPhase,
    totalAnswered,
    scores,
    scorecardNotice,
    questionStartTime,
    tabSwitchCount,
    tabWarning,
  ]);

  // Auto-scroll
  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages, aiLoading, evalLoading]);

  // #9 Anti-cheat: detect tab switching during interview
  useEffect(() => {
    if (screen !== "interview") return;
    const handler = () => {
      if (document.hidden) {
        setTabSwitchCount(c => c + 1);
        setTabWarning("⚠ Tab switch detected — this has been logged in the report.");
        setTimeout(() => setTabWarning(""), 4000);
      }
    };
    document.addEventListener("visibilitychange", handler);
    return () => document.removeEventListener("visibilitychange", handler);
  }, [screen]);

  const addMsg = useCallback((role, text, phase) => {
    setMessages(prev => [...prev, { role, text, phase, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }]);
    if (role === "ai") setQuestionStartTime(Date.now());
  }, []);

  const startInterview = async () => {
    if (!config.candidateName.trim() || !config.role.trim()) return;
    setError(""); setAiLoading(true); setScreen("interview");
    try {
      const q = await generateOpeningQuestion({ phase: PHASES[0], role: config.role, jd: config.jd, experience: config.experience, resumeText, history: [], lastAnswerQuality: null, candidateName: config.candidateName });
      addMsg("ai", q.trim(), PHASES[0]);
    } catch (e) {
      setError(`Failed to start: ${e.message}`);
      setScreen("landing");
    } finally { setAiLoading(false); }
  };

  const sendAnswer = async () => {
    const trimmed = input.trim();
    if (!trimmed || aiLoading || evalLoading) return;
    if (isLowEffortAnswer(trimmed)) {
      setError("Please provide a more detailed answer (at least a short explanation).");
      return;
    }
    setInput(""); setError("");

    // #5 Capture response time
    const responseTime = questionStartTime ? Math.round((Date.now() - questionStartTime) / 1000) : null;

    const userMsg = { role: "user", text: trimmed, phase: PHASES[phaseIdx], time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) };
    const updatedMsgs = [...messages, userMsg];
    setMessages(updatedMsgs);

    const newTotal = totalAnswered + 1;
    setTotalAnswered(newTotal);

    // Evaluate answer (non-blocking)
    const currentQ = getLastMessageByRole(messages, "ai")?.text || "";
    let evalResult = null;
    setEvalLoading(true);
    if (USE_AI_EVALUATION) {
      try {
        evalResult = await evaluateAnswer({ question: currentQ, answer: trimmed, role: config.role, phase: PHASES[phaseIdx], responseTime });
      } catch (e) {
        console.error("Gemini evaluation failed", e);
      }
    } else {
      evalResult = evaluateAnswerLocally({ question: currentQ, answer: trimmed, role: config.role, phase: PHASES[phaseIdx], responseTime });
    }
    const safeEval = evalResult
      ? evalResult
      : { quality: "average", scores: { correctness: 5, relevance: 5, clarity: 5, depth: 5 }, feedback: "Automatic evaluation failed, so a neutral fallback score was used.", questionRelevance: 5, responseTime };
    setLastEval(safeEval);
    const updatedEvals = [...perAnswerEvals, safeEval];
    setPerAnswerEvals(updatedEvals);
    setEvalLoading(false);

    // Determine next phase
    const newQInPhase = qInPhase + 1;
    let nextPhaseIdx = phaseIdx;
    let nextQInPhase = newQInPhase;
    if (newQInPhase >= QUESTIONS_PER_PHASE) { nextPhaseIdx = phaseIdx + 1; nextQInPhase = 0; }

    // End interview after all questions
    if (newTotal >= totalQ) {
      await doScorecard(updatedMsgs, updatedEvals); return;
    }

    setPhaseIdx(nextPhaseIdx); setQInPhase(nextQInPhase);

    // Generate next question
    setAiLoading(true);
    try {
      const q = await generateInterviewQuestion({ phase: PHASES[nextPhaseIdx], role: config.role, jd: config.jd, experience: config.experience, resumeText, history: updatedMsgs, lastAnswerQuality: safeEval.quality });
      addMsg("ai", q.trim(), PHASES[nextPhaseIdx]);
    } catch (e) {
      setError(`Could not generate next question: ${e.message}. Try sending your answer again.`);
    } finally { setAiLoading(false); }
  };

  const doScorecard = async (msgs, evals) => {
    setScreen("scoring");
    if (USE_AI_FINAL_SCORECARD) {
      try {
        const aiScores = await generateScorecard({ role: config.role, jd: config.jd, experience: config.experience, resumeText, history: msgs, perAnswerEvals: evals, tabSwitchCount });
        if (!aiScores) throw new Error("Invalid scorecard payload");
        setScores(aiScores);
        setScorecardNotice("");
      } catch (e) {
        console.error("Scorecard generation failed", e);
        setScores(null);
        setScorecardNotice("Final evaluation could not be generated from Gemini. Please try again.");
      }
    } else {
      setScores(generateLocalScorecard({ role: config.role, perAnswerEvals: evals, tabSwitchCount }));
      setScorecardNotice("Quota-safe mode: final report generated from local analytics to save API usage.");
    }
    setScreen("scorecard");
  };

  const endEarly = async () => {
    if (totalAnswered < MIN_QUESTIONS_FOR_SCORE) { setError(`Answer at least ${MIN_QUESTIONS_FOR_SCORE} questions before ending.`); return; }
    await doScorecard(messages, perAnswerEvals);
  };

  const exportReport = () => {
    const report = buildExportReportPayload();
    if (!report) {
      setError("Gemini-based final report is unavailable. Generate the scorecard again.");
      return;
    }
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url;
    const safeName = (config.candidateName || "candidate").trim().replace(/\s+/g, "_");
    a.download = `interview_${safeName}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const exportReportPdf = () => {
    const report = buildExportReportPayload();
    if (!report) {
      setError("Gemini-based final report is unavailable. Generate the scorecard again.");
      return;
    }
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const pageW = doc.internal.pageSize.getWidth();
    const pageH = doc.internal.pageSize.getHeight();
    const margin = 42;
    const lineH = 16;
    const maxW = pageW - margin * 2;
    let y = margin;

    const ensureSpace = (needed = lineH) => {
      if (y + needed > pageH - margin) {
        doc.addPage();
        y = margin;
      }
    };

    const addTitle = (txt) => {
      ensureSpace(24);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(14);
      doc.text(txt, margin, y);
      y += 22;
    };

    const addLine = (txt) => {
      doc.setFont("helvetica", "normal");
      doc.setFontSize(11);
      const lines = doc.splitTextToSize(String(txt), maxW);
      lines.forEach(line => {
        ensureSpace();
        doc.text(line, margin, y);
        y += lineH;
      });
    };

    addTitle("InterviewSim Final Report");
    addLine(`Candidate: ${report.candidate.candidateName || "N/A"}`);
    addLine(`Role: ${report.candidate.role || "N/A"}`);
    addLine(`Experience: ${report.candidate.experience || "N/A"}`);
    addLine(`Generated At: ${new Date(report.meta.generatedAt).toLocaleString()}`);
    y += 6;

    addTitle("Scores");
    addLine(`Candidate Overall: ${report.scores.candidateScore.overall}/100`);
    addLine(`Interviewer Overall: ${report.scores.interviewerScore.overall}/100`);
    addLine(`Suitability: ${report.scores.hiringRecommendation.suitabilityScore}/100 (${report.scores.hiringRecommendation.verdict})`);
    y += 6;

    addTitle("Interview Duration Analytics");
    const rt = report.analytics.responseTime;
    if (rt) {
      addLine(`Average response time: ${rt.average}s`);
      addLine(`Fastest answer: ${rt.fastest}s`);
      addLine(`Slowest answer: ${rt.slowest}s`);
    } else {
      addLine("Response-time data unavailable.");
    }
    y += 6;

    addTitle("Skill Coverage");
    report.analytics.skillsEvaluated.forEach(skill => addLine(`- ${skill}`));
    y += 6;

    addTitle("Strengths");
    report.scores.strengths.forEach(item => addLine(`- ${item}`));
    y += 6;

    addTitle("Improvements");
    report.scores.improvements.forEach(item => addLine(`- ${item}`));
    y += 6;

    addTitle("Recommendation");
    addLine(report.scores.hiringRecommendation.reason || "No recommendation reason provided.");

    const safeName = (config.candidateName || "candidate").trim().replace(/\s+/g, "_");
    doc.save(`interview_${safeName}_${Date.now()}.pdf`);
  };

  const restart = () => {
    setScreen("landing"); setMessages([]); setPerAnswerEvals([]); setLastEval(null);
    setInterviewerNotes(""); setInput(""); setPhaseIdx(0); setQInPhase(0);
    setTotalAnswered(0); setScores(null); setError("");
    setScorecardNotice("");
    setResumeText(""); setResumeFileName("");
    setTabSwitchCount(0); setTabWarning(""); setQuestionStartTime(null);
    setConfig({ candidateName: "", role: "", jd: "", experience: "2-5" });
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
  };

  const handleKey = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendAnswer(); } };
  const currentAiQ = getLastMessageByRole(messages, "ai")?.text || "";
  const baseScore = calcBaseScore(perAnswerEvals);

  if (screen === "scoring") return (
    <div className="app"><style>{CSS}</style>
      <div className="z1" style={{ position: "fixed", top: "1rem", right: "1rem" }}>
        <button className="theme-btn" onClick={toggleTheme}>{theme === "dark" ? "Light" : "Dark"} Mode</button>
      </div>
      <div className="lscr z1">
        <div className="spin" />
        <p>Analysing transcript — generating dual scorecards…</p>
        <p style={{ opacity: .55 }}>This may take 10–20 seconds</p>
      </div>
    </div>
  );

  if (screen === "scorecard") return (
    <div className="app"><style>{CSS}</style>
      <div className="z1" style={{ position: "fixed", top: "1rem", right: "1rem" }}>
        <button className="theme-btn" onClick={toggleTheme}>{theme === "dark" ? "Light" : "Dark"} Mode</button>
      </div>
      {scorecardNotice && (
        <div className="z1" style={{ width: "100%", display: "flex", justifyContent: "center", paddingTop: "1.5rem" }}>
          <div style={{ maxWidth: 900, width: "calc(100% - 2rem)" }}>
            <ErrBox msg={scorecardNotice} />
          </div>
        </div>
      )}
      <Scorecard scores={scores} config={config} messages={messages} perAnswerEvals={perAnswerEvals} tabSwitchCount={tabSwitchCount} onRestart={restart} onExport={exportReport} onExportPdf={exportReportPdf} />
    </div>
  );

  if (screen === "interview") {
    const isInterviewerMode = mode === "interviewer";
    const showPanel = isInterviewerMode;
    return (
      <div className="app"><style>{CSS}</style>
        <div className={`iroot z1 ${showPanel ? "with-panel" : ""}`}>
          {/* LEFT SIDEBAR */}
          <aside className="sidebar">
            <div className="sbrand">InterviewSim</div>
            <div className={`mind-mode ${mode === "interviewer" ? "int-m" : "cand-m"}`}>
              {mode === "interviewer" ? "👤 Interviewer View" : "🎯 Candidate View"}
            </div>
            <div className="ccrd">
              <div className="cn">{config.candidateName}</div>
              <div className="cr">{config.role}</div>
              <div className="cx">{config.experience} yrs experience</div>
            </div>
            <div>
              <div className="slbl">Interview Phases</div>
              <div className="plist">
                {PHASES.map((p, i) => (
                  <div key={p} className={`prow ${i === phaseIdx ? "on" : i < phaseIdx ? "dn" : "nd"}`}>
                    <div className="pdot" />{p}
                    {i < phaseIdx && <span className="tag ts" style={{ marginLeft: "auto", fontSize: ".58rem" }}>✓</span>}
                  </div>
                ))}
              </div>
            </div>
            <div className="qprog">Progress: <b>{totalAnswered}</b> / {totalQ} questions</div>
            {lastEval && isInterviewerMode && (
              <div className="leval">
                <div className="leval-title">Last Answer Eval</div>
                <div style={{ marginBottom: ".55rem" }}><QTag quality={lastEval.quality} /></div>
                {[["Correctness", (lastEval.scores?.correctness ?? 0) * 10], ["Relevance", (lastEval.scores?.relevance ?? 0) * 10], ["Clarity", (lastEval.scores?.clarity ?? 0) * 10], ["Depth", (lastEval.scores?.depth ?? 0) * 10]].map(([l, v]) => (
                  <SBar key={l} label={l} value={v} />
                ))}
                {lastEval.feedback && <div className="mindfb">{lastEval.feedback}</div>}
              </div>
            )}
            <div style={{ marginTop: "auto", display: "flex", flexDirection: "column", gap: ".55rem" }}>
              {tabSwitchCount > 0 && isInterviewerMode && (
                <div style={{ background: "rgba(248,113,113,.08)", border: "1px solid rgba(248,113,113,.2)", borderRadius: "var(--radius)", padding: ".55rem .75rem", fontSize: ".68rem", color: "var(--danger)", fontFamily: "'DM Mono',monospace" }}>
                  ⚠ Tab switches: {tabSwitchCount}
                </div>
              )}
              {perAnswerEvals.length > 0 && isInterviewerMode && (
                <div style={{ background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: ".55rem .75rem", fontSize: ".68rem", fontFamily: "'DM Mono',monospace", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ color: "var(--muted2)" }}>Base Score</span>
                  <span style={{ color: barColor(baseScore), fontWeight: 700 }}>{baseScore}/100</span>
                </div>
              )}
              {error && <ErrBox msg={error} onDismiss={() => setError("")} />}
              <button className="btn bda bsm bfw" onClick={endEarly} disabled={aiLoading || totalAnswered < MIN_QUESTIONS_FOR_SCORE}>
                End & Generate Report
              </button>
              {totalAnswered < MIN_QUESTIONS_FOR_SCORE && (
                <div style={{ fontSize: ".63rem", color: "var(--muted)", textAlign: "center", fontFamily: "'DM Mono',monospace" }}>Min {MIN_QUESTIONS_FOR_SCORE} answers required</div>
              )}
            </div>
          </aside>

          {/* MAIN CHAT */}
          <div className="mcol">
            <div className="tbar">
              <div className="tbar-l">
                <div className="ldot" />
                <div className="tbar-title">Live Interview — {PHASES[phaseIdx]} Phase</div>
              </div>
              <div className="tbar-r">
                <span className="tag ta">{totalAnswered}/{totalQ} Q</span>
                {!isInterviewerMode && <span className="tag tm">Scoring Hidden</span>}
                <button className="theme-btn" onClick={toggleTheme}>{theme === "dark" ? "Light" : "Dark"} Mode</button>
                <div className="smtoggle">
                  <button className={`smbtn ${mode === "candidate" ? "on" : ""}`} onClick={() => setMode("candidate")}>Candidate</button>
                  <button className={`smbtn ${mode === "interviewer" ? "on" : ""}`} onClick={() => setMode("interviewer")}>Interviewer</button>
                </div>
              </div>
            </div>

            <div className="cfeed" ref={chatRef}>
              {/* Tab-switch warning toast */}
              {tabWarning && (
                <div style={{ background: "rgba(248,113,113,.1)", border: "1px solid rgba(248,113,113,.25)", borderRadius: "var(--radius)", padding: ".6rem 1rem", fontSize: ".75rem", color: "var(--danger)", fontFamily: "'DM Mono',monospace", animation: "fadeUp .2s ease", flexShrink: 0 }}>
                  {tabWarning}
                </div>
              )}
              {messages.length === 0 && !aiLoading && (
                <div style={{ textAlign: "center", color: "var(--muted)", fontSize: ".78rem", fontFamily: "'DM Mono',monospace", padding: "2rem" }}>Interview starting…</div>
              )}
              {messages.map((m, i) => {
                const userIdx = messages.slice(0, i + 1).filter(x => x.role === "user").length - 1;
                const ev = m.role === "user" && perAnswerEvals[userIdx] ? perAnswerEvals[userIdx] : null;
                return (
                  <div key={i} className={`msg ${m.role}`}>
                    <div className="mav">{m.role === "ai" ? "AI" : config.candidateName.charAt(0).toUpperCase()}</div>
                    <div className="mbody">
                      {m.role === "ai" && <div className="mphase"><span className="tag ta">{m.phase}</span></div>}
                      <div className="mbub">{m.text}</div>
                      {ev && isInterviewerMode && (
                        <div className="evalpill">
                          <QTag quality={ev.quality} />
                          <div className="epdims">
                            {[["C", ev.scores?.correctness], ["R", ev.scores?.relevance], ["Cl", ev.scores?.clarity], ["D", ev.scores?.depth]].map(([lbl, val]) => (
                              <div key={lbl} className="edim">
                                <span className="dn">{lbl}</span>
                                <span className="dv" style={{ color: barColor((val ?? 0) * 10) }}>{val ?? "?"}/10</span>
                              </div>
                            ))}
                            <div className="edim">
                              <span className="dn">Q.Rel</span>
                              <span className="dv" style={{ color: barColor((ev.questionRelevance ?? 0) * 10) }}>{ev.questionRelevance ?? "?"}/10</span>
                            </div>
                            {ev.responseTime != null && (
                              <div className="edim">
                                <span className="dn">⏱</span>
                                <span className="dv" style={{ color: "var(--muted2)" }}>{ev.responseTime}s</span>
                              </div>
                            )}
                          </div>
                          {ev.feedback && <div className="epfb">{ev.feedback}</div>}
                        </div>
                      )}
                      <div className="mtime">{m.time}</div>
                    </div>
                  </div>
                );
              })}
              {(aiLoading || evalLoading) && (
                <div className="msg ai">
                  <div className="mav">AI</div>
                  <div className="mbody">
                    <div className="mbub"><div className="tdots"><span /><span /><span /></div></div>
                    <div className="mtime">
                      {evalLoading && !aiLoading
                        ? isInterviewerMode
                          ? "Evaluating response…"
                          : "Reviewing answer…"
                        : "Generating question…"}
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="idock">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder={aiLoading ? "Waiting for question…" : "Type your answer…"}
                disabled={aiLoading}
                rows={1}
              />
              <button className="sbtn" onClick={sendAnswer} disabled={!input.trim() || aiLoading || evalLoading}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
            <div className="hint">Enter to send · Shift+Enter for new line</div>
          </div>

          {/* RIGHT INTERVIEWER PANEL */}
          {showPanel && (
            <aside className="ipanel">
              <div className="iphdr">Current Question</div>
              <div className="iqbox">{currentAiQ || "Waiting for first question…"}</div>
              <div>
                <div className="iphdr" style={{ marginBottom: ".55rem" }}>Private Notes</div>
                <div className="inotes">
                  <textarea value={interviewerNotes} onChange={e => setInterviewerNotes(e.target.value)} placeholder="Add private notes here…" />
                </div>
              </div>
              {perAnswerEvals.length > 0 && (
                <div>
                  <div className="iphdr" style={{ marginBottom: ".55rem" }}>Answer History (last 4)</div>
                  <div className="ahist">
                    {perAnswerEvals.slice(-4).map((e, i) => {
                      const absIdx = perAnswerEvals.length - perAnswerEvals.slice(-4).length + i;
                      const aiMsgs = messages.filter(m => m.role === "ai");
                      return (
                        <div key={i} className="ahrow">
                          <div className="ahq">Q{absIdx + 1}: {(aiMsgs[absIdx]?.text || "").slice(0, 50)}{(aiMsgs[absIdx]?.text || "").length > 50 ? "…" : ""}</div>
                          <div className="ahscores">
                            <QTag quality={e.quality} />
                            <span className="tag tm" style={{ fontSize: ".6rem" }}>Q.Rel {e.questionRelevance}/10</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </aside>
          )}
        </div>
      </div>
    );
  }

  // LANDING
  return (
    <div className="app"><style>{CSS}</style>
      <div className="landing z1">
        <div style={{ position: "absolute", top: "1rem", right: "1rem" }}>
          <button className="theme-btn" onClick={toggleTheme}>{theme === "dark" ? "Light" : "Dark"} Mode</button>
        </div>
        <div className="hero-badge">Interview Simulation Platform</div>
        <h1 className="htitle">Enterprise<br /><span className="grad">Interview Sim</span></h1>
        <p className="hdesc">AI-powered interview simulation — adaptive questions, live response evaluation, and dual scorecards for candidate + interviewer.</p>
        <div className="scard">
          <div className="scard-title">Configure Session</div>
          <div className="mtoggle">
            <button className={`mbtn ${mode === "candidate" ? "on" : ""}`} onClick={() => setMode("candidate")}>🎯 Candidate Mode</button>
            <button className={`mbtn ${mode === "interviewer" ? "on" : ""}`} onClick={() => setMode("interviewer")}>👤 Interviewer Mode</button>
          </div>
          <div className="frow">
            <div className="field">
              <label>Candidate Name *</label>
              <input type="text" placeholder="e.g. Arjun Sharma" value={config.candidateName} onChange={e => setConfig(p => ({ ...p, candidateName: e.target.value }))} />
            </div>
            <div className="field">
              <label>Experience Level</label>
              <select value={config.experience} onChange={e => setConfig(p => ({ ...p, experience: e.target.value }))}>
                <option value="0-1">0–1 yrs (Fresher)</option>
                <option value="2-5">2–5 yrs (Mid)</option>
                <option value="5-10">5–10 yrs (Senior)</option>
                <option value="10+">10+ yrs (Lead)</option>
              </select>
            </div>
          </div>
          <div className="field">
            <label>Role / Position *</label>
            <input type="text" placeholder="e.g. Backend Engineer, Product Manager, Data Scientist" value={config.role} onChange={e => setConfig(p => ({ ...p, role: e.target.value }))} />
          </div>
          <div className="field">
            <label>Job Description <span style={{ color: "var(--muted)", fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>(optional — improves question quality)</span></label>
            <textarea placeholder="Paste the JD or key skills/requirements…" value={config.jd} onChange={e => setConfig(p => ({ ...p, jd: e.target.value }))} />
          </div>
          <div className="field">
            <label>Resume / CV <span style={{ color: "var(--muted)", fontWeight: 400, textTransform: "none", letterSpacing: 0 }}>(optional — personalises questions)</span></label>
            <input
              id="resume-file"
              className="resume-input"
              type="file"
              accept=".txt,.pdf,.docx"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                if (!file) return;
                setResumeFileName(file.name);
                const text = await extractResumeText(file);
                setResumeText(text);
              }}
            />
            <label htmlFor="resume-file" className={`resume-drop ${resumeFileName ? "has-file" : ""}`}>
              {resumeFileName
                ? `✓ ${resumeFileName} — resume loaded`
                : "Click to upload .txt · .pdf · .docx"}
            </label>
            {resumeFileName && (
              <div style={{ marginTop: ".4rem", fontSize: ".68rem", color: "var(--muted2)", fontFamily: "'DM Mono',monospace", display: "flex", justifyContent: "space-between" }}>
                <span>{resumeText.length} characters extracted</span>
                <span style={{ cursor: "pointer", color: "var(--danger)" }} onClick={() => { setResumeText(""); setResumeFileName(""); }}>✕ Remove</span>
              </div>
            )}
          </div>
          {error && <div style={{ marginBottom: ".75rem" }}><ErrBox msg={error} onDismiss={() => setError("")} /></div>}
          <button className="btn bp bfw" onClick={startInterview} disabled={!config.candidateName.trim() || !config.role.trim() || aiLoading}>
            {aiLoading ? "Starting…" : "Begin Interview →"}
          </button>
        </div>
        <div className="hfeats">
          <span>✦ 5 structured phases</span>
          <span>✦ Resume-aware questions</span>
          <span>✦ Live response eval</span>
          <span>✦ Dual scorecard</span>
          <span>✦ Time tracking</span>
          <span>✦ Anti-cheat</span>
        </div>
      </div>
    </div>
  );
}
