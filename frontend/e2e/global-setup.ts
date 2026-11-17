import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

/** Start every run from empty per-user tables in the *test* database (the script refuses any other database). */
export default function globalSetup() {
  const py = process.env.E2E_PYTHON
    ?? (process.platform === "win32" ? path.join(ROOT, ".venv", "Scripts", "python.exe") : path.join(ROOT, ".venv", "bin", "python"));
  const db = process.env.E2E_DATABASE_URL ?? "postgresql://postgres:postgres@localhost:5433/carwale_test";
  execFileSync(py, ["-m", "tests.fakes.reset_test_db", db], { cwd: ROOT, stdio: "inherit" });
}
