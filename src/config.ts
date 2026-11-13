/**
 * ConvoAI - Environment & Deployment Configuration
 *
 * Separates development defaults from production hardening while
 * keeping local startup effortless and cloud hosting configurable.
 */

import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, "..");

export type RuntimeEnvironment = "development" | "production" | "test";

export interface AppConfig {
  nodeEnv: RuntimeEnvironment;
  isProduction: boolean;
  host: string;
  port: number;
  databasePath: string;
  corsOrigins: string[];
  trustProxy: boolean | number;
  rateLimitPerMinute: number;
}

function parseEnvironment(raw?: string): RuntimeEnvironment {
  const val = (raw ?? "development").trim().toLowerCase();
  if (val === "production") return "production";
  if (val === "test") return "test";
  return "development";
}

function parsePositiveInt(raw: string | undefined, fallback: number): number {
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

function parseTrustProxy(
  raw: string | undefined,
  isProduction: boolean
): boolean | number {
  if (raw === undefined || raw.trim() === "") {
    return isProduction ? 1 : false;
  }
  const normalized = raw.trim().toLowerCase();
  if (normalized === "true") return true;
  if (normalized === "false" || normalized === "0") return false;
  const hops = Number(normalized);
  if (Number.isInteger(hops) && hops > 0) return hops;
  return isProduction ? 1 : false;
}

function parseCorsOrigins(
  raw: string | undefined,
  isProduction: boolean
): string[] {
  const parsed = (raw ?? "")
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);

  if (isProduction && parsed.includes("*")) {
    console.warn(
      "[convoai] Wildcard CORS_ORIGINS='*' is ignored in production. Configure explicit origin URLs."
    );
  }

  return parsed.filter((origin) => {
    if (isProduction && origin === "*") {
      return false;
    }
    return origin === "*" || /^https?:\/\/[^/\s]+$/i.test(origin);
  });
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  const nodeEnv = parseEnvironment(env.NODE_ENV);
  const isProduction = nodeEnv === "production";
  const host = (env.HOST ?? "0.0.0.0").trim() || "0.0.0.0";
  const port = parsePositiveInt(env.PORT, 3000);

  const rawDbPath = (env.DATABASE_PATH ?? "").trim();
  const databasePath = rawDbPath
    ? rawDbPath === ":memory:"
      ? ":memory:"
      : path.resolve(rawDbPath)
    : path.join(ROOT_DIR, "convoai.db");

  const rateLimitPerMinute =
    env.RATE_LIMIT_PER_MINUTE !== undefined && env.RATE_LIMIT_PER_MINUTE.trim() !== ""
      ? Math.max(0, Math.floor(Number(env.RATE_LIMIT_PER_MINUTE) || 0))
      : 30;

  return {
    nodeEnv,
    isProduction,
    host,
    port,
    databasePath,
    corsOrigins: parseCorsOrigins(env.CORS_ORIGINS, isProduction),
    trustProxy: parseTrustProxy(env.TRUST_PROXY, isProduction),
    rateLimitPerMinute,
  };
}
