import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const fixtureRoot = fileURLToPath(new URL("../", import.meta.url));
const modulePath = fileURLToPath(new URL("../src/math.js", import.meta.url));
const childProgram = `
import { pathToFileURL } from "node:url";
const [modulePath, exportName, serializedArguments] = process.argv.slice(1);
const target = await import(pathToFileURL(modulePath).href);
const callable = target[exportName];
if (typeof callable !== "function") throw new Error("requested export is not callable");
const result = await callable(...JSON.parse(serializedArguments));
process.stdout.write(JSON.stringify({ result }));
`;

function invokeInChild(exportName, arguments_) {
  const child = spawnSync(
    process.execPath,
    [
      "--input-type=module",
      "--eval",
      childProgram,
      modulePath,
      exportName,
      JSON.stringify(arguments_),
    ],
    {
      cwd: fixtureRoot,
      encoding: "utf8",
      env: { LANG: "C", PATH: process.env.PATH ?? "" },
      maxBuffer: 64 * 1024,
      timeout: 5_000,
    },
  );
  assert.equal(child.signal, null, child.stderr || "child process was terminated");
  assert.equal(child.status, 0, child.stderr || "child process failed");
  const payload = JSON.parse(child.stdout);
  assert.deepEqual(Object.keys(payload), ["result"]);
  return payload.result;
}

test("adds two numbers", () => {
  assert.equal(invokeInChild("add", [2, 3]), 5);
});

test("multiplies two numbers", () => {
  assert.equal(invokeInChild("multiply", [3, 4]), 12);
  assert.equal(invokeInChild("multiply", [-2, 5]), -10);
  assert.equal(invokeInChild("multiply", [0, 8]), 0);
});
