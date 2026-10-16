import {createRequire as __createRequire} from 'node:module';const require=__createRequire(import.meta.url);

// scripts/intelligence-universe.mjs
import { readFile, writeFile, mkdir, rename } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { pathToFileURL } from "node:url";
function directoryStocks(content, market) {
  const lines = content.trim().split(/\r?\n/), columns = lines.shift().split("|");
  if (!columns.includes("Security Name") || !columns.includes("Test Issue")) throw new Error("INVALID_DIRECTORY");
  return lines.filter((line) => !line.startsWith("File Creation Time:")).map((line) => Object.fromEntries(line.split("|").map((value, index) => [columns[index], value]))).filter((row) => row["Test Issue"] === "N" && (market === "NASDAQ" || row.Exchange === "N")).map((row) => ({ symbol: row.Symbol || row["ACT Symbol"], name: row["Security Name"], aliases: [row["Security Name"].split(" - ")[0].replace(/\b(Inc\.?|Corporation|Corp\.?|Limited|Ltd\.?)\b/gi, "").trim()].filter((name) => name.length >= 4), market, sector: null, exposures: [] }));
}
async function main() {
  const catalog = JSON.parse(await readFile(new URL("../src/data/securities.json", import.meta.url), "utf8"));
  const stocks = catalog.securities.filter((row) => /\.(TW|TWO)$/.test(row.symbol)).map((row) => ({ symbol: row.symbol.replace(/\.(TW|TWO)$/, ""), name: row.name, market: row.symbol.endsWith(".TWO") ? "TPEx" : "TWSE", sector: null, exposures: [] }));
  const sources = [{ market: "TWSE", url: "https://openapi.twse.com.tw/", asOf: catalog.updatedAt }, { market: "TPEx", url: "https://www.tpex.org.tw/", asOf: catalog.updatedAt }];
  for (const [market, file] of [["NASDAQ", "nasdaqlisted.txt"], ["NYSE", "otherlisted.txt"]]) {
    const url = "https://www.nasdaqtrader.com/dynamic/SymDir/" + file;
    const response = await fetch(url, { redirect: "error", signal: AbortSignal.timeout(3e4) });
    if (!response.ok) throw new Error("DIRECTORY_UNAVAILABLE");
    const content = await response.text();
    if (content.length > 1e7) throw new Error("DIRECTORY_TOO_LARGE");
    stocks.push(...directoryStocks(content, market));
    sources.push({ market, url, asOf: (/* @__PURE__ */ new Date()).toISOString() });
  }
  if (process.env.INTELLIGENCE_PROFILES_FILE) {
    const profiles = JSON.parse(await readFile(resolve(process.env.INTELLIGENCE_PROFILES_FILE), "utf8"));
    if (!Array.isArray(profiles)) throw new Error("INVALID_PROFILES");
    const byKey = new Map(profiles.map((profile) => [profile.market + ":" + profile.symbol, profile]));
    for (const stock of stocks) {
      const profile = byKey.get(stock.market + ":" + stock.symbol);
      if (profile) {
        stock.sector = profile.sector || null;
        stock.exposures = profile.exposures || [];
        stock.aliases = profile.aliases || [];
      }
    }
  }
  const output = resolve(process.env.INTELLIGENCE_UNIVERSE_FILE || "intelligence-runtime/universe.json");
  const result = { asOf: (/* @__PURE__ */ new Date()).toISOString(), sources, stocks };
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output + ".tmp", JSON.stringify(result));
  await rename(output + ".tmp", output);
  console.log(JSON.stringify({ stocks: stocks.length, markets: [...new Set(stocks.map((stock) => stock.market))], profiled: stocks.filter((stock) => stock.exposures.length).length }));
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) main().catch(() => {
  console.error("UNIVERSE_REFRESH_FAILED: previous universe retained");
  process.exitCode = 1;
});
export {
  directoryStocks
};
